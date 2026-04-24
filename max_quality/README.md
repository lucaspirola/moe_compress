# Strategy A — Maximum Quality MoE Compression for Qwen3.6-35B-A3B @ 30%

Implementation of Strategy A from
[`pirola/moe-compression-workflow/VALIDATED_STRATEGIES.md`](https://huggingface.co/pirola/moe-compression-workflow/blob/main/VALIDATED_STRATEGIES.md),
targeting **Qwen/Qwen3.6-35B-A3B** with **30% total parameter reduction** (expert
pruning + non-uniform SVD compounding).

Plan file: `~/.claude/plans/using-https-huggingface-co-pirola-moe-co-mutable-galaxy.md`

## Pipeline

| Stage | Module | What it does | A100 cost |
|-------|--------|--------------|-----------|
| 0 | `stage0_super_experts` | `down_proj` max-activation → blacklist | ~2 min |
| 1 | `stage1_grape` | per-layer redundancy → non-uniform expert budgets | ~5 min |
| 2 | `stage2_reap_ream` | REAP scoring + REAM merge (sequential) | ~1.5 h |
| 3 | `stage3_svd` | D-Rank + Swift-SVD+ + AA-SVD + block refine | ~45 min |
| 4 | `stage4_eora` | training-free low-rank compensation | ~5 min |
| 5 | `stage5_router_kd` | router-only KL distillation | ~20 min |
| 6 | `stage6_validate` | WikiText-2 PPL + zero-shot + gen | ~10 min |

## Quick start (A100 80 GB)

```bash
source /home/lucas/ai/venv/bin/activate
pip install -r requirements.txt
python -m moe_compress.run_pipeline \
    --config configs/qwen36_35b_a3b_30pct.yaml \
    --model Qwen/Qwen3.6-35B-A3B \
    --artifacts-dir ./artifacts \
    --target-ratio 0.30
```

Resume from a specific stage (e.g. after an OOM mid-Stage-3):

```bash
python -m moe_compress.run_pipeline --config ... --resume-from-stage 3
```

## Local smoke test (RTX 5080, 16 GB)

```bash
pytest tests/ -v                         # synthetic MoE unit tests
pytest tests/test_smoke_qwen3_0_5b.py    # end-to-end on a small MoE model
```

## Protected components (never touched by pruning/SVD)

- Shared expert at every MoE layer
- Attention weights (DeltaNet and full-attention projections)
- Embeddings, `lm_head`, layer norms
- Router weights (except Stage 5, which updates *only* these)
- Super experts on the Stage 0 blacklist

## Risk register

| Risk | Level | Mitigation |
|------|-------|------------|
| REAM + variable-N′_l interaction | med-high | Per-layer MSE monitor in Stage 2; bump budget 10% on outlier |
| AA-SVD on MoE weights | medium | SVD limited to expert matrices; block-level L-BFGS refine |
| Full multi-stage pipeline untested | high | Per-stage checkpointing; Stage 6 hard gate on quality metrics |
| DeltaNet hybrid attention | unknown | All attention weights frozen across all stages |

## Success criteria (Stage 6 vs uncompressed)

- WikiText-2 PPL: ≤ +3% relative
- ARC-C / HellaSwag: ≤ 1.5 pp absolute drop
- HumanEval / MATH-500: ≤ 3 pp absolute drop
- Actual param reduction: ≥ 30.0%
