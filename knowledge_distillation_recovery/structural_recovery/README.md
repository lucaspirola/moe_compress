# Chapter 1 — Structural Recovery at BF16

Logit-only forward KLD distillation (Minitron, arxiv:2407.14679) on the BF16
artifact produced by `../../max_quality`. Recovers quality lost to expert
pruning + SVD + EoRA + Router KD before any quantization.

Spec: [`pirola/knowledge-distillation-recovery`](https://huggingface.co/pirola/knowledge-distillation-recovery)
— Chapter 1 of `QUALITY_RECOVERY_GUIDE.md` plus `VRAM_OPTIMIZATION.md`
(Appendix D).

Plan: `~/.claude/plans/on-another-session-we-soft-pixel.md`.

## Stack (Appendix D.2)

| Component                | This pipeline                          |
|--------------------------|----------------------------------------|
| Teacher                  | `Qwen/Qwen3.6-35B-A3B-FP8` (~37 GB)    |
| Student                  | `stage5_final/` from max_quality (BF16)|
| Optimizer                | 8-bit AdamW (`bnb.optim.AdamW8bit`)    |
| Loss                     | Forward KLD only, T = 1.0              |
| Activations              | Gradient checkpointing                 |
| Parallelism              | DeepSpeed ZeRO-3                       |
| Hardware                 | HF Jobs `a100x4` (4× A100, 320 GB)     |

## Quick start

Smoke (1× H200, ~30 min, ~$2.50):

```bash
SMOKE=1 STUDENT_REPO=pirola/qwen3-6-35b-a3b-strategy-a-30pct-<ts> \
    ./hf_jobs/submit.sh
```

a100-large (1× A100-80GB) is **NOT** viable — the FP8 teacher (~37 GB) +
BF16 student (~70 GB) total 107 GB, which exceeds 80 GB. The smoke tier
uses h200 (141 GB) for headroom.

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
accelerate launch --config_file ds_configs/zero3_offload_optim.json \
    -m structural_recovery.run_recovery \
    --config configs/qwen36_35b_a3b_chapter1_smoke.yaml \
    --student /path/to/stage5_final \
    --artifacts-dir ./recovery_artifacts
```

## Output

`artifacts/chapter1_recovered/` — sharded safetensors + `compressed_metadata.json`
with `pipeline_stage="chapter1_recovered"`. Loads via
`moe_compress.utils.model_io.load_compressed_model(...)`.
