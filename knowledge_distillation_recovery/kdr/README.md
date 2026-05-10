# kdr — Knowledge Distillation Recovery

Unified Chapter-1 BF16 KD + Chapter-3 Deployment-Aware QAD trainer for the
`moe_compress` pipeline. One mode flag, one FKLD loss, asymmetric K/V quant
recipes, compressed-tensors output. See
[`/home/lucas/.claude/plans/radiant-pondering-beacon.md`](../../../.claude/plans/radiant-pondering-beacon.md)
for the full architecture and `../requirements/` for the HLR/LLR set.

## Status

Phase 2 (skeleton) — `pyproject.toml` scaffolding, typed Pydantic schemas, and
stub modules raising `NotImplementedError`. Real behaviour lands in Phases 3+.

## Modes

| Mode      | What it does                                                                |
|-----------|-----------------------------------------------------------------------------|
| `bf16`    | Plain forward-KLD logit distillation (Chapter 1). No `mtq.quantize` call.   |
| `da_qad`  | Chapter 3 DA-QAD: `mtq.quantize` installs fake-quant per the YAML's `quant` block; same FKLD loop. |

## Output

HF compressed-tensors safetensors loadable via:

```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("path/to/kdr_output", trust_remote_code=True)
```

The input student's `compressed_metadata.json` (MoE-factored topology) is
preserved verbatim if present; omitted otherwise.

## Out of scope

GGUF (llama.cpp) and MLX (jangq) deployment targets are out of scope for v0.
Users targeting those re-quantize from the BF16 master in the kdr output and
accept the recovery loss; the `LlamaCppBackend` and `JangqBackend` plugins are
v1+ if ever.
