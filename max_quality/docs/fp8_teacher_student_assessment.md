# FP8 model variant — impact assessment

**Status:** assessment only, no code changes.
**Date:** 2026-04-25
**Question asked:** Should we swap `Qwen/Qwen3.6-35B-A3B` (BF16) for `Qwen/Qwen3.6-35B-A3B-FP8` as either *teacher* (Stage 5 KD) or *student* (full pipeline)? Specifically, would it free VRAM, allow larger batches or a smaller GPU tier, and speed up the run?

**Short answer:** **Stay on BF16 + A100.** On the current Ampere hardware FP8 is auto-dequantised at load time and provides zero VRAM, batch-size, or speed benefit. FP8 only pays off on Hopper (H100/H200), and even there it benefits only the teacher role and the post-pipeline final artifact — not the SVD-heavy core stages.

---

## What `Qwen/Qwen3.6-35B-A3B-FP8` actually is

Source: model card https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8 + `hf models info` + transformers source (`quantizer_finegrained_fp8.py`).

| Property | Value |
| --- | --- |
| FP8 dtype | `torch.float8_e4m3fn` |
| Quantisation scheme | HF "fine-grained FP8", **block-wise scaling 128 × 128** |
| `quantization_config.quant_method` | `"fp8"` |
| Coverage | only MoE expert linear projections (~96% of params). Norms, routing gates, attention projections, embeddings, MTP layers stay BF16. |
| On-disk size | 35.95 GB (BF16 baseline: ~36 GB — **essentially identical**, the unquantized 4% offsets the 50% saving on the rest) |

**Implication for downloads / bucket cache:** none. Same number of bytes on disk.

---

## Hard blocker on A100: auto-dequantisation at load

`transformers/src/transformers/quantizers/quantizer_finegrained_fp8.py` requires CUDA compute capability **≥ 8.9** for native FP8.

| GPU | Architecture | Compute capability | Native FP8? |
| --- | --- | --- | --- |
| A100-80GB *(current)* | Ampere | 8.0 | **No** — quantizer flips `dequantize=True` |
| H100-80GB | Hopper | 9.0 | Yes |
| H200 | Hopper | 9.0 | Yes |
| L40S | Ada Lovelace | 8.9 | Yes |

When the FP8 model is loaded on A100, `from_pretrained` materialises every FP8 weight as **BF16 in GPU memory**. Consequences on the current `--flavor a100-large` HF Jobs flavor:

| Metric | BF16 baseline | FP8 on A100 (auto-dequant) | Δ |
| --- | --- | --- | --- |
| Model resident memory | ~70.3 GB | ~72 GB | ~0 (slightly worse) |
| Free GPU memory | ~5.4 GB | ~5.4 GB | 0 |
| Effective Stage 2 `batch_size` | 4 | 4 | 0 |
| Stage 2 wall time / layer | ~6 min | ~6 min (likely +small dequant overhead) | 0 to slightly slower |
| Minimum GPU tier | A100-80GB | A100-80GB | 0 |
| Stage 2 OOM risk on `lm_head` 7.58 GB allocation | gated by `expandable_segments:True` | unchanged | 0 |

**Verdict on Ampere:** **FP8 buys nothing on A100 and may slightly regress** due to the load-time dequant pass and possibly less optimal kernel paths.

---

## What changes on H100 / H200 (cc ≥ 9.0)

| Metric | BF16 on H100 | FP8 on H100 |
| --- | --- | --- |
| Model resident memory | ~72 GB | ~36 GB |
| Free GPU memory | ~8 GB | ~44 GB |
| Plausible Stage 2 `batch_size` | 4 (same lm_head bottleneck) | 16–32 |
| Smallest GPU tier that fits | 80 GB | 40 GB (e.g. L40S, half H100) |
| FP8 GEMM speedup vs BF16 | n/a | ~2× **only on GEMM-bound stages** |

**Stage-by-stage speedup analysis on Hopper:**

| Stage | Hot path | Benefits from FP8? |
| --- | --- | --- |
| 0 — Super-expert detection | calibration forwards | Yes (forward GEMMs) |
| 1 — GRAPE | weight similarity in float32 | No (no forward) |
| 2 — REAP / REAM | calibration forwards + linalg | Partial — forwards yes, REAM cost in fp32 |
| 3 — SVD | `torch.linalg.svd / cholesky / solve_triangular` in **float32** | **No — float32 linalg, no FP8 kernel** |
| 4 — EoRA | float32 SVD on residuals | No |
| 5 — Router KD | teacher + student forwards, AdamW on `gate.weight` | Yes (forwards) |
| 6 — Eval | calibration forwards | Yes |

The pipeline's wall clock is dominated by **Stage 3 SVD (float32, no FP8 kernel exists)** and Stage 2 sequential per-layer profiling. FP8's headline ~2× GEMM speedup helps Stages 0/2/5/6 but does not shorten Stages 1/3/4. Net wall-clock improvement on Hopper is meaningful but not 2×.

**Cost trade-off:** HF Jobs Hopper hourly is significantly higher than A100. Whether the wall-clock improvement pays for it requires a per-stage runtime breakdown of the current run before deciding.

---

## Pipeline-internal dtype audit (FP8 doesn't break math)

Every dtype-sensitive operation in the codebase already up-casts to float32 before the numerically-fragile step. Audited paths:

| Stage | Hot operation | File:line | Status |
| --- | --- | --- | --- |
| 0 | `down_proj.amax()` for super-expert detection | `utils/activation_hooks.py:76` | Clean — runs on the loaded dtype (BF16 after auto-dequant) |
| 1 | Pairwise cosine over flattened weights | `stage1_grape.py:74` | Clean — explicit `.to(torch.float32)` |
| 2 | REAM cost matrix (cosine of weights + router) | `stage2_reap_ream.py:215, 225` | Clean |
| 2 | Frequency-weighted merge | `stage2_reap_ream.py:282-284` | Clean (avg in fp32; write-back via `bank.set` casts to target dtype) |
| 2 | Input covariance accumulation | `utils/activation_hooks.py:180-181` | Clean (fp32 accum, configurable storage dtype) |
| 3 | `svdvals`, `svd`, `cholesky`, `cholesky_inverse`, `solve_triangular` | `stage3_svd.py:186, 249-262` | Clean — all on fp32 (no FP8 linalg in PyTorch) |
| 3 | L-BFGS refine | `stage3_svd.py:340-363` | Clean (fp32 optimisation; final cast to factor dtype) |
| 4 | EoRA SVD on residuals + `eigh` | `stage4_eora.py:126-135` | Clean |
| 5 | KD KL loss | `stage5_router_kd.py:112-118` | Clean — both teacher and student logits cast to fp32 before softmax/KL |

**No FP8-specific blocker exists inside our code.** PyTorch has no FP8 SVD/Cholesky kernels, but we never hit them — every linalg site already up-casts.

---

## Latent bug only relevant on Hopper (where FP8 stays packed)

If FP8 is preserved in memory rather than auto-dequantised, Stage 2's `_merge_experts_inplace` writes fp32-averaged weights via `ExpertMatrixBank.set` → `target.copy_(W.to(dtype=target.dtype))` (`utils/model_io.py:314`).

For an FP8 stacked tensor with companion **block-wise scales**, naively casting fp32 → e4m3fn ignores the per-block scales and writes numerically wrong values relative to the original quantised representation. The merged experts would be mis-scaled.

This is a real bug **only if** we ever hold FP8 stacked tensors in `bank.set`'s target.
- On A100 (auto-dequant) it never happens; targets are BF16. **Today's setup is safe.**
- On H100 with native FP8 retention, we'd need to either (a) recompute block scales after each merge write, or (b) keep stacked tensors in BF16 throughout the pipeline and only use FP8 for the teacher's read-only forward.

---

## Recommendation

### Stay on BF16 + A100 for the current run

The current pipeline is correctly tuned and FP8 buys nothing on Ampere.

### If/when the user moves to Hopper, the cleanest setup is

1. **Teacher (Stage 5 only)** — load `Qwen/Qwen3.6-35B-A3B-FP8` directly. It's read-only; the quantizer materialises it natively on Hopper. Frees ~36 GB during KD which lets student + teacher comfortably co-reside on a single 80 GB GPU and removes the need for any sharding logic.
2. **Student (Stages 0–4)** — keep BF16 throughout. This sidesteps the merge-quantisation-scale latent bug entirely and lets all the float32-cast linalg paths stay as-is. SVD wall time doesn't shrink under FP8 anyway, so there is no gain from running the student in FP8 during compression.
3. **Final artifact** — quantise the post-Stage-5 BF16 checkpoint to FP8 as a separate post-process (HF `optimum-fp8` / `llm-compressor`). Compression and quantisation are **orthogonal**: there is no benefit to running compression *on* FP8 weights, only to *producing* an FP8 final artifact.

### Defer the FP8 swap entirely on the current run

Finish Stages 3–6 on BF16 + A100 first. Revisit FP8 only as a cost-optimisation for follow-up runs, and only if moving the job to Hopper.

---

## Critical files referenced

- `/home/lucas/ai/moe_compress/max_quality/configs/qwen36_35b_a3b_30pct.yaml` — `model.torch_dtype: bfloat16`
- `/home/lucas/ai/moe_compress/max_quality/hf_jobs/entrypoint.py` — `torch>=2.5,<2.11`, `--flavor a100-large`
- `/home/lucas/ai/moe_compress/max_quality/src/moe_compress/utils/model_io.py:314` — `ExpertMatrixBank.set` casts to target dtype (the Hopper-only concern)
- `/home/lucas/ai/moe_compress/max_quality/src/moe_compress/stage5_router_kd.py:47-55` — Teacher load path; the one place where teacher dtype is independent of student dtype today

---

## Verification plan (only if FP8 is later adopted on Hopper)

1. **Load probe (cheap)** — HF Job on `--flavor h100-small` running:
   ```python
   from transformers import AutoModelForCausalLM
   m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.6-35B-A3B-FP8")
   print(m.model.language_model.layers[0].mlp.experts.gate_up_proj.dtype)
   ```
   Expect `torch.float8_e4m3fn`. If it prints `bfloat16`, the auto-dequant path was taken anyway and FP8 buys nothing on this flavor.
2. **Forward parity** — same prompt through BF16 vs FP8 teacher; cosine of logits ≥ 0.999. If not, FP8 weight loading dropped scales somewhere.
3. **Stage 5 only** — one-stage run with student=BF16 + teacher=FP8 vs student=BF16 + teacher=BF16; KD loss curves should match within ~1%.
4. **End-to-end PPL** — WikiText-2 ppl delta between FP8-teacher KD and BF16-teacher KD; expect within 0.5%.

---

## Sources

- [Qwen/Qwen3.6-35B-A3B-FP8 model card](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8)
- [transformers fine-grained FP8 quantizer source](https://github.com/huggingface/transformers/blob/main/src/transformers/quantizers/quantizer_finegrained_fp8.py)
- [transformers quantization docs — fine-grained FP8](https://huggingface.co/docs/transformers/quantization/finegrained_fp8)
- [NVIDIA A100 datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf) — confirms no FP8 tensor cores on cc 8.0
