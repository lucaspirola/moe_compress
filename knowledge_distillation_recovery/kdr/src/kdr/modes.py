"""Training mode flag for kdr (LLR-0006).

The single binary mode flag controlling whether kdr runs Chapter-1 BF16 KD only
(`bf16`) or Chapter-3 deployment-aware QAD (`da_qad`). Every other knob is a
property of the YAML's `quant` block, not the mode.
"""

from typing import Literal

Mode = Literal["bf16", "da_qad"]
"""The two modes kdr supports.

* `bf16`: structural-recovery-style logit-only forward-KLD distillation. No
  fake-quant is installed; the student remains in BF16 throughout.
* `da_qad`: deployment-aware QAD per the Ch3 spec. `mtq.quantize` installs
  fake-quant per the YAML's `quant` block before the first training step.
"""
