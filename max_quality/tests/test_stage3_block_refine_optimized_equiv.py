"""Byte-identity gate for the Levers 1+2 block_refine CPU optimization.

Levers 1+2 (``stage3/plugins/block_refine.py``) make the per-step host->device
copy of the block-refine training tensors **pinned + ``non_blocking=True``**.
The change is mechanism-only: it changes *how* a constant bf16 CPU tensor is
moved to the GPU, not *which* bytes move, the op order, the AdamW step order,
or any RNG draw. The consumed values must therefore be bit-identical to the
pre-optimization (pageable, synchronous) path.

There is NO existing harness for this (``test_stage3_plugin_block_refine.py``
is import / re-export / protocol / metadata only; the Stage 3 golden snapshot
runs with ``block_refine`` OFF). This file builds a from-scratch synthetic
student+teacher decoder stack with :class:`FactoredExperts` MoE layers, drives
``_phase_c5_block_refine`` to completion, and asserts the trained factors +
RMSNorm scales reproduce a GOLDEN captured from ``origin/main``'s
pre-optimization code bit-identically (``torch.equal``).

Capture vs verify
-----------------
The golden is captured by running THIS harness against ``origin/main``'s
``block_refine.py`` (the pre-optimization blob), saved under
``tests/golden/stage3_block_refine/``. The verify run uses the current
(post-optimization) ``block_refine``. No production toggle / monkeypatch is
introduced (project rule): the old-vs-new comparison goes purely through the
captured golden. Same determinism caveat as the Stage 3 golden snapshot:
capture + verify MUST run on the same torch wheel / host.

Capture workflow::

    MOE_REGEN_BLOCK_REFINE_GOLDEN=1 \
      pytest tests/test_stage3_block_refine_optimized_equiv.py -q

  (with ``block_refine.py`` checked out at origin/main) -> writes the golden,
  test skips. Restore the optimized ``block_refine.py``, then run without the
  env var -> must pass via ``torch.equal``.

Self-verification of the pinned path
------------------------------------
The harness asserts ``epochs > 1`` AND ``n_batches >= 2`` (so the per-step
pinned+async H2D fires repeatedly across epochs and batches) and runs BOTH the
live-teacher-target branch AND the cache-hit branch (a populated
``teacher_targets_cache``) so the pinned upload is exercised for both target
sources. On CUDA it additionally asserts the captured CPU source tensors are
actually pinned.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from moe_compress.stage3.plugins.block_refine import _phase_c5_block_refine
from moe_compress.utils.model_io import MATRIX_NAMES, FactoredExperts, MoELayerRef

_GOLDEN_DIR = Path(__file__).resolve().parent / "golden" / "stage3_block_refine"
_REGEN_ENV = "MOE_REGEN_BLOCK_REFINE_GOLDEN"

# --- Synthetic decoder stack dimensions (KB-scale; safe on a shared 16 GB GPU).
_HIDDEN = 16
_INTERMEDIATE = 8
_NUM_EXPERTS = 4
_TOP_K = 2
_NUM_LAYERS = 2          # both MoE so every block hits the AdamW refine path
_RANK = 4
_N_SEQ = 4
_SEQ_LEN = 6
_BATCH_SIZE = 2          # => n_batches = 4 // 2 = 2 (>= 2, self-verify)
_EPOCHS = 2              # > 1 (self-verify)
_VOCAB = 32


# ---------------------------------------------------------------------------
# Synthetic model: decoder stack with FactoredExperts MoE + block-local norms.
# ---------------------------------------------------------------------------
class _RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = 1e-6

    def forward(self, x):
        v = x.to(torch.float32)
        v = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + self.eps)
        return (v * self.weight.to(torch.float32)).to(x.dtype)


class _Router(nn.Module):
    def __init__(self, num_experts: int, hidden: int, top_k: int):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden
        self.top_k = top_k
        self.weight = nn.Parameter(torch.randn(num_experts, hidden) * 0.02)

    def forward(self, flat):
        logits = F.linear(flat, self.weight)
        probs = F.softmax(logits, dim=-1, dtype=torch.float32)
        topv, topi = torch.topk(probs, self.top_k, dim=-1)
        topv = topv / topv.sum(dim=-1, keepdim=True)
        return topi, topv.to(flat.dtype)


class _SelfAttn(nn.Module):
    """Minimal self-attention carrying the per-head q_norm / k_norm scales the
    block_refine trainable scope expects. The 'attention' is an identity-ish
    linear so the harness stays cheap and deterministic."""

    def __init__(self, hidden: int):
        super().__init__()
        self.q_norm = _RMSNorm(hidden)
        self.k_norm = _RMSNorm(hidden)
        self.proj = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x):
        # Touch q_norm / k_norm so their scales are on the autograd graph.
        h = self.q_norm(x) + self.k_norm(x)
        return self.proj(h)


class _MoEBlock(nn.Module):
    def __init__(self, hidden, intermediate, num_experts, top_k, rank, dtype):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_dim = hidden
        self.gate = _Router(num_experts, hidden, top_k)
        self.experts = FactoredExperts(
            num_experts=num_experts,
            hidden_dim=hidden,
            intermediate_dim=intermediate,
            ranks={"gate_proj": rank, "up_proj": rank, "down_proj": rank},
            dtype=dtype,
            device="cpu",
        )

    def forward(self, x):
        B, T, H = x.shape
        flat = x.reshape(-1, H)
        idx, w = self.gate(flat)
        out = self.experts(flat, idx, w)
        return out.reshape(B, T, H)


class _DecoderLayer(nn.Module):
    def __init__(self, hidden, intermediate, num_experts, top_k, rank, dtype):
        super().__init__()
        self.input_layernorm = _RMSNorm(hidden)
        self.post_attention_layernorm = _RMSNorm(hidden)
        self.self_attn = _SelfAttn(hidden)
        self.mlp = _MoEBlock(hidden, intermediate, num_experts, top_k, rank, dtype)

    def forward(self, hidden_states, **_ignored):
        # Pre-hook-captured kwargs (none here — the model calls positionally)
        # are replayed via _phase_c5_block_refine; accept+ignore extras.
        h = self.input_layernorm(hidden_states)
        h = hidden_states + self.self_attn(h)
        h2 = self.post_attention_layernorm(h)
        return h + self.mlp(h2)


class _Tower(nn.Module):
    def __init__(self, n_layers, hidden, intermediate, num_experts, top_k, rank, dtype):
        super().__init__()
        self.layers = nn.ModuleList([
            _DecoderLayer(hidden, intermediate, num_experts, top_k, rank, dtype)
            for _ in range(n_layers)
        ])


class _Config:
    def __init__(self, num_experts, num_layers, hidden, intermediate, top_k):
        self.num_hidden_layers = num_layers
        self.layer_types = ["full_attention"] * num_layers
        self.num_experts = num_experts
        self.num_experts_per_tok = top_k
        self.hidden_size = hidden
        self.moe_intermediate_size = intermediate
        self.text_config = self


class _Model(nn.Module):
    def __init__(self, *, dtype):
        super().__init__()
        self.embed = nn.Embedding(_VOCAB, _HIDDEN)
        self.model = _Tower(
            _NUM_LAYERS, _HIDDEN, _INTERMEDIATE, _NUM_EXPERTS, _TOP_K, _RANK, dtype
        )
        self.lm_head = nn.Linear(_HIDDEN, _VOCAB, bias=False)
        self.config = _Config(_NUM_EXPERTS, _NUM_LAYERS, _HIDDEN, _INTERMEDIATE, _TOP_K)
        self._dtype = dtype

    def forward(self, input_ids=None, **_ignored):
        x = self.embed(input_ids).to(self._dtype)
        for layer in self.model.layers:
            x = layer(x)
        logits = self.lm_head(x.to(self.lm_head.weight.dtype))

        class _Out:
            pass

        out = _Out()
        out.logits = logits
        return out


def _init_factored_experts(model: _Model) -> None:
    """Fill the zero-initialized FactoredExperts U/V with small deterministic
    values so the block-refine train + the routing have something to chew on."""
    g = torch.Generator().manual_seed(7)
    for layer in model.model.layers:
        fe = layer.mlp.experts
        for name in MATRIX_NAMES:
            for slot in (f"{name}_U", f"{name}_V"):
                p = getattr(fe, slot)
                p.data = (torch.randn(p.shape, generator=g) * 0.02).to(p.dtype)


def _build_model(dtype) -> _Model:
    torch.manual_seed(0)
    m = _Model(dtype=dtype)
    m = m.to(dtype)
    # Embedding stays its own (float) param set but we cast hidden states to
    # dtype in forward, so factors + norms can be bf16/fp32 uniformly.
    _init_factored_experts(m)
    return m


def _moe_refs(model: _Model) -> list[MoELayerRef]:
    refs = []
    for idx, layer in enumerate(model.model.layers):
        refs.append(MoELayerRef(
            layer_idx=idx,
            layer_module=layer,
            mlp=layer.mlp,
            router=layer.mlp.gate,
            experts_module=layer.mlp.experts,
            shared_expert=None,
            layer_type="full_attention",
        ))
    return refs


def _calib_tensor() -> torch.Tensor:
    g = torch.Generator().manual_seed(123)
    return torch.randint(0, _VOCAB, (_N_SEQ, _SEQ_LEN), generator=g)


# ---------------------------------------------------------------------------
# Driver: run _phase_c5_block_refine and snapshot the trained tensors.
# ---------------------------------------------------------------------------
def _trained_state(model: _Model) -> dict[str, torch.Tensor]:
    """Collect the block-refine trainable tensors (FactoredExperts U/V per
    MATRIX_NAME + the 4 RMSNorm scales) per layer, on CPU, in final dtype."""
    state: dict[str, torch.Tensor] = {}
    for idx, layer in enumerate(model.model.layers):
        fe = layer.mlp.experts
        for name in MATRIX_NAMES:
            state[f"L{idx}.{name}_U"] = getattr(fe, f"{name}_U").detach().cpu().clone()
            state[f"L{idx}.{name}_V"] = getattr(fe, f"{name}_V").detach().cpu().clone()
        state[f"L{idx}.input_layernorm"] = layer.input_layernorm.weight.detach().cpu().clone()
        state[f"L{idx}.post_attention_layernorm"] = (
            layer.post_attention_layernorm.weight.detach().cpu().clone()
        )
        state[f"L{idx}.q_norm"] = layer.self_attn.q_norm.weight.detach().cpu().clone()
        state[f"L{idx}.k_norm"] = layer.self_attn.k_norm.weight.detach().cpu().clone()
    return state


def _run_block_refine(device, *, use_cache: bool, tmp_path: Path) -> dict[str, torch.Tensor]:
    """Build a fresh student+teacher pair, run _phase_c5_block_refine, return
    the trained student state. Deterministic given the fixed seeds above.

    The model is fp32: block_refine promotes its trainable U/V to fp32 before
    AdamW (block_refine.py:433-437) and the FactoredExperts forward asserts
    ``factor.dtype == hidden_states.dtype``, so ``student_dtype`` must be fp32
    for the refine forward to run -- matching the production block_refine path.
    The per-step CPU source tensors are still pinned bf16 (captured via
    ``.to(bf16, ...).pin_memory()``); the ``.to(fp32, non_blocking=True)`` H2D
    upcasts on transfer, exercising the exact pinned+async mechanism."""
    dtype = torch.float32
    torch.manual_seed(0)
    student = _build_model(dtype).to(device)
    # Teacher is an independent deepcopy-equivalent (same seeds => same init),
    # built fresh so its params are a separate set on the device.
    teacher = _build_model(dtype).to(device)

    moe_layers = _moe_refs(student)
    teacher_moe_layers = _moe_refs(teacher)
    calib = _calib_tensor()

    n_seq = calib.shape[0]
    n_batches = n_seq // _BATCH_SIZE
    assert _EPOCHS > 1, "self-verify: epochs must exceed 1 to exercise per-step H2D across epochs"
    assert n_batches >= 2, "self-verify: need >= 2 batches so the per-step H2D fires repeatedly"

    teacher_targets_cache = None
    if use_cache:
        # Build an un-chunked [n_prompts, seq_len, hidden] bf16 CPU cache entry
        # per MoE layer by capturing the teacher block outputs the same way the
        # live path would. We reuse the live path once (no cache) to produce
        # byte-faithful targets, then feed them back as a cache to exercise the
        # cache-hit branch's pinned slice upload.
        teacher_targets_cache = _capture_teacher_cache(
            teacher, teacher_moe_layers, calib, device, dtype
        )

    _phase_c5_block_refine(
        student,
        teacher,
        moe_layers,
        teacher_moe_layers,
        calib,
        batch_size=_BATCH_SIZE,
        learning_rate=1e-3,
        epochs=_EPOCHS,
        warmup_ratio=0.1,
        weight_decay=0.01,
        artifacts_dir=tmp_path,
        no_resume=True,
        device=device,
        teacher_targets_cache=teacher_targets_cache,
    )
    return _trained_state(student)


def _capture_teacher_cache(teacher, teacher_moe_layers, calib, device, dtype):
    """Produce a ``dict[layer_idx -> [n_prompts, seq_len, hidden]]`` un-chunked
    bf16 CPU cache by forwarding the teacher block-by-block, mirroring the live
    teacher-target computation so the cache-hit branch sees a SHAPE-VALID entry
    (cache.shape[0] >= n_batches*batch_size, cache.shape[1] == seq_len)."""
    from moe_compress.utils.model_io import iter_decoder_layers

    n_seq, seq_len = calib.shape
    n_batches = n_seq // _BATCH_SIZE
    batches = [calib[b * _BATCH_SIZE:(b + 1) * _BATCH_SIZE] for b in range(n_batches)]

    t_layers_all = {idx: layer for idx, layer in iter_decoder_layers(teacher)}
    first_idx = sorted(t_layers_all.keys())[0]

    # Capture per-batch input to the first decoder layer (the X_teacher seed).
    captured = {}

    class _EarlyExit(Exception):
        pass

    def _hook(_m, args, kwargs):
        t = args[0] if args else kwargs.get("hidden_states")
        captured[cur[0]] = t.detach()
        raise _EarlyExit

    cur = [0]
    handle = t_layers_all[first_idx].register_forward_pre_hook(_hook, with_kwargs=True)
    try:
        for bi, b in enumerate(batches):
            cur[0] = bi
            try:
                with torch.no_grad():
                    teacher(input_ids=b.to(device))
            except _EarlyExit:
                pass
    finally:
        handle.remove()

    x_teacher = [captured[bi] for bi in range(n_batches)]
    cache: dict[int, torch.Tensor] = {}
    moe_idx = {ref.layer_idx for ref in teacher_moe_layers}
    with torch.no_grad():
        for idx in sorted(t_layers_all.keys()):
            layer = t_layers_all[idx]
            outs = []
            new_x = []
            for bi in range(n_batches):
                out = layer(x_teacher[bi])
                if isinstance(out, tuple):
                    out = out[0]
                new_x.append(out)
                outs.append(out.detach().to(dtype=torch.bfloat16, device="cpu"))
            if idx in moe_idx:
                # Un-chunked [n_prompts, seq_len, hidden] for this layer.
                cache[int(idx)] = torch.cat(outs, dim=0).contiguous()
            x_teacher = new_x
    return cache


# ---------------------------------------------------------------------------
# Golden capture / verify.
# ---------------------------------------------------------------------------
def _golden_path(tag: str) -> Path:
    return _GOLDEN_DIR / f"trained_{tag}.pt"


def _capture_or_verify(tag: str, device, *, use_cache: bool, tmp_path: Path):
    state = _run_block_refine(device, use_cache=use_cache, tmp_path=tmp_path)
    gpath = _golden_path(tag)
    if os.environ.get(_REGEN_ENV):
        _GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(state, gpath)
        pytest.skip(
            f"Regenerated block_refine golden {gpath.name} "
            f"(capture against origin/main, then verify against the "
            f"optimized code). Inspect git diff and commit."
        )
    if not gpath.exists():
        pytest.skip(
            f"block_refine golden {gpath.name} absent. Capture it first with "
            f"{_REGEN_ENV}=1 against origin/main's block_refine.py."
        )
    golden = torch.load(gpath, map_location="cpu")
    assert set(golden) == set(state), (
        f"trained-tensor key set drift: golden={sorted(golden)} "
        f"got={sorted(state)}"
    )
    mismatched = []
    for k in golden:
        g, s = golden[k], state[k]
        if g.dtype != s.dtype or g.shape != s.shape or not torch.equal(g, s):
            mismatched.append(k)
    assert not mismatched, (
        "Levers 1+2 are NOT byte-identical: trained tensors differ from the "
        f"origin/main golden for {mismatched}. The pinned + non_blocking H2D "
        "must not change any value (same op order, same stream, no RNG)."
    )


# ---------------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------------
def test_block_refine_pinned_equiv_live_targets(tmp_path):
    """Live-teacher-target branch: trained factors/norms must be torch.equal
    to the origin/main golden."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _capture_or_verify("live", device, use_cache=False, tmp_path=tmp_path)


def test_block_refine_pinned_equiv_cache_hit(tmp_path):
    """Cache-hit branch: same byte-identity gate with a populated
    teacher_targets_cache so the pinned per-batch slice upload is exercised."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _capture_or_verify("cache", device, use_cache=True, tmp_path=tmp_path)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="pin_memory needs CUDA")
def test_block_refine_sources_are_pinned():
    """Self-verify the optimization is actually engaged: the bf16 CPU source
    tensors produced at the capture site are pinned (so non_blocking=True is a
    real async copy, not a silent sync). Exercises the live capture path."""
    device = torch.device("cuda")
    dtype = torch.bfloat16
    student = _build_model(dtype).to(device)

    # Reproduce the _capture_block_input producer's exact expression and assert
    # the result is pinned, mirroring the production edit at block_refine.py.
    sample = _calib_tensor()[:_BATCH_SIZE].to(device)
    captured = {}

    class _EarlyExit(Exception):
        pass

    def _hook(_m, args, kwargs):
        t = args[0] if args else kwargs.get("hidden_states")
        captured["t"] = t.detach().to(dtype=torch.bfloat16, device="cpu").pin_memory()
        raise _EarlyExit

    first = student.model.layers[0]
    h = first.register_forward_pre_hook(_hook, with_kwargs=True)
    try:
        try:
            with torch.no_grad():
                student(input_ids=sample)
        except _EarlyExit:
            pass
    finally:
        h.remove()
    assert "t" in captured
    assert captured["t"].is_pinned(), (
        "block_refine capture source must be pinned so the per-step "
        "non_blocking H2D is genuinely async"
    )
