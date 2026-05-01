# H200 + HF Jobs Truth Reference

Canonical hardware and HF Jobs facts for the spec-compliance audit.
Each line is a self-contained fact citing its source. Subsequent
crossref agents will grep this file by fact substring.

## NVIDIA H200 GPU — hardware

- HBM3e capacity is **141 GB** per H200 GPU. <!-- source: https://www.nvidia.com/en-us/data-center/h200/ fetched 2026-05-01 -->
- HBM3e memory bandwidth is **4.8 TB/s** per H200 GPU. <!-- source: https://www.nvidia.com/en-us/data-center/h200/ fetched 2026-05-01 -->
- H200 is based on the **NVIDIA Hopper architecture**, compute capability **SM_90**. <!-- source: https://www.nvidia.com/en-us/data-center/h200/ fetched 2026-05-01 -->
- H200 supports **FP8 Tensor Cores** (E4M3 + E5M2 formats), inherited from Hopper. <!-- source: https://www.nvidia.com/en-us/data-center/h200/ fetched 2026-05-01 -->
- FP8 is **Hopper-only** (H100 / H200 / B100); **A100 (Ampere) does NOT support FP8** natively — loading an FP8 checkpoint on A100 either fails or dequantizes to BF16 with zero VRAM savings. <!-- source: /home/lucas/.claude/projects/-home-lucas-ai-moe-compress/memory/feedback_fp8_a100_hardware.md -->
- H200 SXM FP8 Tensor Core throughput: **3,958 TFLOPS** (with sparsity; dense ≈ half). <!-- source: https://www.nvidia.com/en-us/data-center/h200/ fetched 2026-05-01 -->
- H200 SXM BF16 Tensor Core throughput: **1,979 TFLOPS**. <!-- source: https://www.nvidia.com/en-us/data-center/h200/ fetched 2026-05-01 -->
- H200 SXM FP16 Tensor Core throughput: **1,979 TFLOPS** (matches BF16). <!-- source: https://www.nvidia.com/en-us/data-center/h200/ fetched 2026-05-01 -->
- H200 SXM TF32 Tensor Core throughput: **989 TFLOPS** (TF32 is natively supported by Hopper Tensor Cores). <!-- source: https://www.nvidia.com/en-us/data-center/h200/ fetched 2026-05-01 -->
- H200 SXM FP32 (non–Tensor-Core) throughput: **67 TFLOPS** — i.e. FP32 has dramatically reduced throughput vs BF16/FP16/TF32 Tensor Core paths; numerical kernels on H200 should prefer BF16 / TF32 when possible. <!-- source: https://www.nvidia.com/en-us/data-center/h200/ fetched 2026-05-01 -->
- H200 supports INT8 Tensor Core operations. <!-- source: https://www.nvidia.com/en-us/data-center/h200/ fetched 2026-05-01 -->
- H200 supports FP64 Tensor Core operations. <!-- source: https://www.nvidia.com/en-us/data-center/h200/ fetched 2026-05-01 -->

## H200 software stack — attention kernels and torch.compile

- **FlashAttention-2** runs on Hopper (SM_90) GPUs including H200 — it is the baseline for FA-3. <!-- source: https://github.com/Dao-AILab/flash-attention (FA repo states FA2 supports Ampere/Ada/Hopper); cross-checked NVIDIA Hopper spec at https://www.nvidia.com/en-us/data-center/h200/ fetched 2026-05-01 -->
- **FlashAttention-3 is Hopper-specific** — it exploits SM_90 features (warp-specialization, asynchronous TMA, FP8 Tensor Cores) and therefore runs on H100 / H200 but **not on A100**. <!-- source: https://tridao.me/blog/2024/flash3/ (Tri Dao, "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision") fetched 2026-05-01 -->
- `torch.compile` works on H200 for **prefill-dominant / static-shape graphs**; dynamic-shape graphs (variable seq-len, KV-cache decode) trigger graph recompiles and can degrade or hang under cudagraphs. Use `dynamic=True` or `mode="reduce-overhead"` carefully, and prefer padding to bucketed shapes for cudagraph capture. <!-- source: PyTorch torch.compile cudagraph notes https://pytorch.org/docs/stable/torch.compiler_cudagraph_trees.html fetched 2026-05-01; also AUDIT_PROMPT.md §03_truth_refs guidance -->
- Cudagraphs (`mode="reduce-overhead"`) require static input shapes, no CPU-GPU sync inside the graph, and no in-place ops on captured tensors — common pitfalls when wrapping HF generate / decode loops on H200. <!-- source: https://pytorch.org/docs/stable/torch.compiler_cudagraph_trees.html fetched 2026-05-01 -->

## HF Jobs — flavors, host RAM, timeout

- HF Jobs **default timeout is 30 minutes**; jobs without an explicit `--timeout` flag are killed at 30 min with `status=ERROR, message="Job timeout"`. <!-- source: https://huggingface.co/docs/hub/jobs-pricing fetched 2026-05-01 -->
- Auto-memory confirms the 30-min default and warns: "Treat 'Job timeout' as equivalent to 'cancelled mid-run' for durability planning." <!-- source: /home/lucas/.claude/projects/-home-lucas-ai-moe-compress/memory/feedback_hf_jobs_timeout.md -->
- Standard HF Jobs **`h200` flavor (1× H200)**: **23 vCPU, 256 GB host RAM, 141 GB GPU VRAM, $5.00/hr**. <!-- source: https://huggingface.co/docs/hub/jobs-pricing fetched 2026-05-01 -->
- HF Jobs **`h200x2` (2× H200)**: 46 vCPU, **512 GB** host RAM, 282 GB GPU VRAM, $10.00/hr. <!-- source: https://huggingface.co/docs/hub/jobs-pricing fetched 2026-05-01 -->
- HF Jobs **`h200x4` (4× H200)**: 92 vCPU, **1024 GB** host RAM, 564 GB GPU VRAM, $20.00/hr. <!-- source: https://huggingface.co/docs/hub/jobs-pricing fetched 2026-05-01 -->
- HF Jobs **`h200x8` (8× H200)**: 184 vCPU, **2048 GB** host RAM, 1128 GB GPU VRAM, $40.00/hr. <!-- source: https://huggingface.co/docs/hub/jobs-pricing fetched 2026-05-01 -->
- HF Jobs timeout supports unit suffixes (`s`/`m`/`h`/`d`); always pass an explicit `--timeout` sized to workload + headroom (e.g. 8h per heavy stage, 12–16h multi-stage). <!-- source: https://huggingface.co/docs/huggingface_hub/guides/jobs fetched 2026-05-01; cross-ref /home/lucas/.claude/projects/-home-lucas-ai-moe-compress/memory/feedback_hf_jobs_timeout.md -->

## HF Jobs — durability of bucket FUSE vs Hub commits

- HF Jobs **bucket volume mounts use hf-mount semantics**: writes are buffered (streaming default = in-memory until `close()`; `--advanced-writes` = staged-to-disk with debounced 2–30 s flush window). **A crash, cancel, or timeout before flush → data loss.** <!-- source: /home/lucas/.claude/projects/-home-lucas-ai-moe-compress/memory/reference_hf_jobs_durability.md (researched 2026-04-25 from HF storage-buckets / hf-mount docs) -->
- `hf jobs cancel` SIGKILLs the pod with **no documented grace period**; treat as worst-case (FUSE write-back caches discarded). <!-- source: /home/lucas/.claude/projects/-home-lucas-ai-moe-compress/memory/reference_hf_jobs_durability.md -->
- HF Jobs **bucket FUSE is non-durable on cancel/timeout**: writes that the in-pod process believes succeeded may never reach the bucket S3 store. Real incident 2026-04-25: Stage 2 logged "complete" but post-cancel `hf buckets ls` showed only stale prior-run artifacts. <!-- source: /home/lucas/.claude/projects/-home-lucas-ai-moe-compress/memory/feedback_hf_jobs_bucket_cancel.md -->
- **Hub commits via `HfApi.upload_folder` / `upload_large_folder` are durable** the moment the HTTP commit returns 200 OK. `upload_large_folder` is resumable via local `./cache/huggingface/` checkpoints (hash, preupload, commit each checkpointed). <!-- source: /home/lucas/.claude/projects/-home-lucas-ai-moe-compress/memory/reference_hf_jobs_durability.md -->
- Canonical durability rule: **bucket = scratch space only; each pipeline stage that produces a durable artifact must `upload_large_folder` to a Hub repo before the job exits**, and resume must download from the Hub repo, not the bucket. <!-- source: /home/lucas/.claude/projects/-home-lucas-ai-moe-compress/memory/reference_hf_jobs_durability.md -->
- Multi-stage pipelines should run **`STOP_AFTER_STAGE=N` per stage with one job per stage**, each ending in a Hub upload — Hub commit is the durability boundary. <!-- source: /home/lucas/.claude/projects/-home-lucas-ai-moe-compress/memory/reference_hf_jobs_durability.md -->
