# kdr — Knowledge Distillation Recovery

Unified Chapter-1 BF16 KD + Chapter-3 Deployment-Aware QAD trainer for the
`moe_compress` pipeline. One mode flag, one FKLD loss, asymmetric K/V quant
recipes, compressed-tensors output. See
[`/home/lucas/.claude/plans/radiant-pondering-beacon.md`](../../../.claude/plans/radiant-pondering-beacon.md)
for the full architecture and `../requirements/` for the HLR/LLR set.

## Status

| Phase                                         | State                                       |
|-----------------------------------------------|---------------------------------------------|
| 0 — `req` Python `#` scanner patch            | ✅ Upstreamed                               |
| 1 — Requirements (12 HLRs, 49 LLRs)           | ✅ Authored                                 |
| 2 — Skeleton (Pydantic, mypy strict, ruff)    | ✅ Landed                                   |
| 3a — FKLD numerical parity                    | ✅ Bit-equal vs structural_recovery         |
| 3b — BF16 loop on stand-ins                   | ✅ Code-port complete                       |
| 4 — QuantBackend layer                        | ✅ ModelOpt + Native + factory + save_kdr_artifact |
| 5 — Mode integration + ZAYA1 adapter          | ✅ All adapter methods + router replay      |
| 6 — Vast.ai docker bootstrap                  | ✅ bootstrap.sh + run_id + HF Hub upload    |
| 7.1 — ZAYA1-8B `bf16` parity smoke            | ⏸️ Awaiting vast.ai 1× H200 / A100-80GB   |
| 7.2 — ZAYA1-8B `da_qad` smoke (NVFP4 + INT4 KV) | ⏸️ Awaiting vast.ai                       |
| 7.3 — Compressed-tensors round-trip           | ⏸️ Awaiting Phase 7.2 output               |
| 7.4 — RTX 5080 NVFP4 deployment validation    | ⏸️ Awaiting RTX 5080 hardware acquisition  |
| 8 — Cutover (SUPERSEDED-BY tagging)           | ⏸️ Gated on 7.1 sign-off                   |

`mypy --strict` clean (31 source files); `ruff` clean; `pytest` 176/176;
`req coverage` 37/49 LLRs implemented + tested. Remaining unimplemented LLRs
(LLR-0008/0034/0044/0046/0047) are CLI polish + Phase 7 hardware gates.

## Modes

| Mode      | What it does                                                                |
|-----------|-----------------------------------------------------------------------------|
| `bf16`    | Plain forward-KLD logit distillation (Chapter 1). No `mtq.quantize` call.   |
| `da_qad`  | Chapter 3 DA-QAD: `mtq.quantize` installs fake-quant per the YAML's `quant` block; same FKLD loop, plus router replay (LLR-0025) for MoE QAD stability. |

## Output

HF compressed-tensors safetensors loadable via:

```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("path/to/kdr_output", trust_remote_code=True)
```

The input student's `compressed_metadata.json` (MoE-factored topology) is
preserved verbatim if present; omitted otherwise.

## Validation runs (Phase 7)

Phase 7 is real-hardware validation. The kdr code is feature-complete (Phases
0–6 closed), but the static-analysis surface (mypy/ruff/pytest) cannot detect
GPU-side regressions. Phase 7 runs are required before SUPERSEDED-BY tagging
of `structural_recovery` (Phase 8 cutover).

| Run    | Hardware              | What it proves                                                       |
|--------|-----------------------|----------------------------------------------------------------------|
| 7.1    | 1× H200 OR A100-80GB  | `bf16` 200-step parity smoke; PPL within 0.5% of `structural_recovery` |
| 7.2    | 1× H200 OR A100-80GB  | `da_qad` 50M-token smoke; no NaN; FKLD monotone-decreasing           |
| 7.3    | Any GPU               | `AutoModelForCausalLM.from_pretrained(kdr_out)` round-trip + PPL match |
| 7.4    | RTX 5080 (Blackwell)  | Real-hardware NVFP4 deployment vs simulated training-time PPL        |

Launch via `docker/bootstrap.sh` on a vast.ai instance — see
[`docker/README.md`](docker/README.md) for the operator runbook.

## Cutover (Phase 8)

After Phase 7.1 closes, `structural_recovery` is marked
`SUPERSEDED-BY: kdr` (project-level note in `../requirements/` since
`structural_recovery` is outside the `.req` graph). The `structural_recovery`
files are RETAINED as historical reference for future audit; they are NOT
deleted.

## Out of scope

GGUF (llama.cpp) and MLX (jangq) deployment targets are out of scope for v0.
Users targeting those re-quantize from the BF16 master in the kdr output and
accept the recovery loss; the `LlamaCppBackend` and `JangqBackend` plugins are
v1+ if ever.

Multi-GPU ZeRO-3 is not validated in Phase 7 (single-GPU is sufficient for
ZAYA1's 8.4 B-total / 760 M-active footprint). The loop dispatches correctly
under DS3 in principle (LLR-0048 call order is preserved), but no run has
been executed against multi-GPU.
