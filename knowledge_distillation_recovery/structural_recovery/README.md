# Chapter 1 — Structural Recovery at BF16

Logit-only forward KLD distillation (Minitron, arxiv:2407.14679) on the BF16
artifact produced by `../../max_quality`. Recovers quality lost to expert
pruning + SVD + EoRA + Router KD before any quantization.

Spec: [`pirola/knowledge-distillation-recovery`](https://huggingface.co/pirola/knowledge-distillation-recovery)
— Chapter 1 of `QUALITY_RECOVERY_GUIDE.md` plus `VRAM_OPTIMIZATION.md`
(Appendix D).

## Stack (Appendix D.2)

| Component                | Light tier (a100x4)                    | Smoke tier (1× H200)              |
|--------------------------|----------------------------------------|-----------------------------------|
| Teacher                  | `Qwen/Qwen3.6-35B-A3B` BF16 (~70 GB)   | `Qwen/Qwen3.6-35B-A3B-FP8` (~37 GB) |
| Student                  | `stage5_final/` from max_quality (BF16)| same                              |
| Optimizer                | DeepSpeedCPUAdam (CPU offload)         | 8-bit AdamW (`bnb.optim.AdamW8bit`)|
| Loss                     | Forward KLD only, T = 1.0              | same                              |
| Activations              | Gradient checkpointing                 | same                              |
| Parallelism              | DeepSpeed ZeRO-3 (4-way)               | none (single GPU)                 |
| Hardware                 | HF Jobs `a100x4` (4× A100, 320 GB)     | HF Jobs `h200` (141 GB)           |

**FP8 teacher is Hopper-only.** A100 has no FP8 tensor cores, so the light
tier on a100x4 must use the BF16 teacher (sharded ~17.5 GB / GPU under
ZeRO-3). The smoke tier on H200 can and does use the FP8 teacher to fit
teacher + student in 141 GB.

## Quick start

Smoke (1× H200, ~30 min, ~$2.50):

```bash
SMOKE=1 STUDENT_REPO=pirola/qwen3-6-35b-a3b-strategy-a-30pct-<ts> \
    ./hf_jobs/submit.sh
```

a100-large (1× A100-80GB) is **NOT** viable. On A100 the teacher must be
BF16 (~70 GB; FP8 needs Hopper) and the student is BF16 (~70 GB) — total
140 GB exceeds 80 GB. The smoke tier uses 1× H200 (141 GB) where the
FP8 teacher (~37 GB) + BF16 student (~70 GB) fits.

Light tier (4× A100, ~6 h, ~$60):

```bash
STUDENT_REPO=pirola/qwen3-6-35b-a3b-strategy-a-30pct-<ts> \
    ./hf_jobs/submit.sh
```

Local equivalent (with shared venv at `/home/lucas/ai/venv`):

```bash
source /home/lucas/ai/venv/bin/activate
pip install -e .
pytest tests/ -v
# Smoke (single-GPU, no DeepSpeed — bnb 8-bit AdamW, matches entrypoint
# behavior under SMOKE=1):
python -m structural_recovery.run_recovery \
    --config configs/qwen36_35b_a3b_chapter1_smoke.yaml \
    --student /path/to/stage5_final \
    --artifacts-dir ./recovery_artifacts --smoke

# Light (multi-GPU under DeepSpeed ZeRO-3):
accelerate launch --use_deepspeed --deepspeed_config_file ds_configs/zero3_offload_optim.json \
    --mixed_precision bf16 \
    -m structural_recovery.run_recovery \
    --config configs/qwen36_35b_a3b_chapter1_light.yaml \
    --student /path/to/stage5_final \
    --artifacts-dir ./recovery_artifacts
```

## Output

`artifacts/chapter1_recovered/` — sharded safetensors + `compressed_metadata.json`
with `pipeline_stage="chapter1_recovered"`. Loads via
`moe_compress.utils.model_io.load_compressed_model(...)`.
