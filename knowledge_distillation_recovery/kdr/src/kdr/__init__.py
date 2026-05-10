"""kdr — Knowledge Distillation Recovery.

Unified Ch1 BF16 KD + Ch3 DA-QAD trainer. Mode flag drives whether `mtq.quantize`
is installed before the FKLD loop. Output is HF compressed-tensors safetensors.

See `/home/lucas/.claude/plans/radiant-pondering-beacon.md` for the architecture
and `knowledge_distillation_recovery/requirements/` for the HLR/LLR set.
"""

__version__ = "0.1.0"
