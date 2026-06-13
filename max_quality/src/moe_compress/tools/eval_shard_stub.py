"""Width-derived stub model + tokenizer for the Stage-6 eval-shard GOLDEN GATE.

Test-support leaf (imported by the golden-gate test AND reconstructed inside the
spawn worker via the ``_STUB_WIDTH_MODEL`` seam). The stub makes the
group-geometry dependence of the gen path PROVABLE: ``generate`` emits a token
sequence encoding ``w = input_ids.shape[1]`` (the padded batch width, exactly
the real ``input_len`` at eval_harness.py:150), which decodes to ``W{w}``. A
group-geometry change is therefore GUARANTEED to flip a completion, so the
negative control (a non-mod-8 split) is non-vacuous by construction.

Pure stdlib + torch; no HF model. Stays in ``tools/`` as a leaf utility.
"""
from __future__ import annotations

import torch


class _WidthStubConfig:
    # _generate_batched asserts eager attention.
    _attn_implementation = "eager"


class _WidthStubModel:
    """Stub whose generate() appends ``max_new`` tokens encoding the padded
    input width. Mirrors model.generate's contract: returns ``[input_ids ;
    new_tokens]`` so ``_generate_batched`` slices ``out[j, input_len:]`` and
    decodes the width tokens."""

    def __init__(self):
        self.config = _WidthStubConfig()

    def parameters(self):
        # _generate_batched infers device from next(parameters()) when device is
        # None; yield a CPU tensor so it lands on CPU.
        yield torch.zeros(1)

    def generate(self, *, input_ids, attention_mask=None, max_new_tokens=4,
                 do_sample=False, pad_token_id=0, **kwargs):
        bsz, width = input_ids.shape
        # Encode the width as a fixed-length run of "width tokens": token id
        # 1000 + digit, one per decimal digit of ``width``, padded/truncated to
        # max_new_tokens. The tokenizer decodes 1000+d -> str(d), and 999 -> "".
        digits = [1000 + int(c) for c in str(width)]
        if len(digits) < max_new_tokens:
            digits = digits + [999] * (max_new_tokens - len(digits))
        else:
            digits = digits[:max_new_tokens]
        new = torch.tensor(digits, dtype=input_ids.dtype).unsqueeze(0).repeat(bsz, 1)
        return torch.cat([input_ids, new], dim=1)


class _WidthStubTokenizer:
    """Char-level stub: each char -> one token id (so prompt length == token
    count, giving deliberately varied widths). pad_to_multiple_of rounds the
    padded width up. decode() turns width tokens (1000+d) into ``W<digits>``."""

    padding_side = "right"
    pad_token_id = 0
    eos_token_id = None

    def __call__(self, prompts, *, return_tensors="pt", padding=True,
                 pad_to_multiple_of=None, truncation=False,
                 add_special_tokens=False):
        # one token per char (id = ord(c) clamped into a small range, >=1)
        seqs = [[max(ord(c) % 900, 1) for c in p] for p in prompts]
        maxlen = max((len(s) for s in seqs), default=0)
        if pad_to_multiple_of:
            rem = maxlen % pad_to_multiple_of
            if rem:
                maxlen += pad_to_multiple_of - rem
        ids, mask = [], []
        for s in seqs:
            padn = maxlen - len(s)
            if self.padding_side == "left":
                row = [self.pad_token_id] * padn + s
                m = [0] * padn + [1] * len(s)
            else:
                row = s + [self.pad_token_id] * padn
                m = [1] * len(s) + [0] * padn
            ids.append(row)
            mask.append(m)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
        }

    def decode(self, ids, skip_special_tokens=True):
        out = []
        for t in ids.tolist():
            if t >= 1000:
                out.append(str(t - 1000))
            # 999 (filler) and anything else -> dropped
        return "W" + "".join(out)


def build_width_stub_model():
    return _WidthStubModel()


def build_width_stub_tokenizer():
    return _WidthStubTokenizer()
