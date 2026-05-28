"""Stage 2 profile-pass `layer_in` hook + writer subscription (CRITICAL-1 / E-1).

Verifies the new wiring landed by ``fix/critical1-vllm-layer-in-hook``:

* Canonical writer's ``_on_layer_in_callback`` grows the per-layer
  reservoir on demand and routes through the vectorised
  :class:`_LayerInputAccumulator` (Plugin #1 Opt-C / Vitter Algorithm R).
* The dump path finalises each accumulator into a bf16 ``[N, hidden]``
  tensor, with the legacy ``(0, 0)`` placeholder retained for layers
  that never saw a token.
* The checkpoint save/load preserves accumulator state
  (``buffer`` + ``seen``) byte-identically across processes (via the
  per-layer-seeded RNG).
* The reader's hidden-bug demote (``cost_alignment='output'`` + empty
  reservoir → ``partial_hit``) fires correctly.

Mirrors plan §7 a / b / e / g (the v2 distill case ships in
``test_expert_distill_v2_populated_reservoir.py``).
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch

from moe_compress.calibration import stage2_profile_writer as s2pw
from moe_compress.pipeline.context import PipelineContext
from moe_compress.stage2.plugins.stage2_profile_cache import (
    Stage2ProfileCacheProvider,
)
from moe_compress.stage2.profiling import _LayerInputAccumulator
from moe_compress.utils.activation_hooks import (
    InputCovarianceAccumulator,
    ReamCostAccumulator,
)
from moe_compress.utils.cached_calibration_signals import (
    SCHEMA_VERSIONS,
    Stage2ProfilePayloadV3,
    load_stage2_profile_v3,
)


def _jsonl(tmp_path: Path) -> Path:
    return tmp_path / "calib" / "self_traces.jsonl"


# ---------------------------------------------------------------------------
# §7.a — writer callback handler smoke + reservoir growth.
# ---------------------------------------------------------------------------
def test_layer_in_callback_lazy_constructs_accumulator():
    """First call creates a per-layer accumulator and stores the batch."""
    s2pw._reset_state_for_tests()
    s2pw.setup(
        cov_storage_dtype="float16",
        layer_idx_to_rank={3: 0},
    )

    hidden = torch.randn(16, 32, dtype=torch.bfloat16)
    s2pw._on_layer_in_callback(layer_idx=3, hidden=hidden)

    state = s2pw._get_state()
    assert 3 in state.layer_input_reservoir
    acc = state.layer_input_reservoir[3]
    assert isinstance(acc, _LayerInputAccumulator)
    assert acc.buffer is not None
    assert acc.buffer.shape == (16, 32)
    assert acc.seen == 16


def test_layer_in_callback_accumulates_across_batches():
    """Multiple sub-cap batches concatenate into a single reservoir buffer."""
    s2pw._reset_state_for_tests()
    s2pw.setup(
        cov_storage_dtype="float16",
        layer_idx_to_rank={0: 0},
    )

    for _ in range(3):
        s2pw._on_layer_in_callback(
            layer_idx=0,
            hidden=torch.randn(16, 32, dtype=torch.bfloat16),
        )

    acc = s2pw._get_state().layer_input_reservoir[0]
    # Default _LAYER_INPUT_MAX_SAMPLES = 8192 >> 48; all three batches
    # fit below the cap → buffer grows to 48.
    assert acc.buffer.shape == (48, 32)
    assert acc.seen == 48


def test_layer_in_callback_per_layer_seed_determinism():
    """Two writer instances with the same layer_idx contract → identical buffers.

    Drives the accumulator past its capacity (default 8192) by patching
    the module-level constant locally so we don't have to feed millions
    of tokens. Restores the constant in a finally block so this test
    can't bleed into later tests via ordering.
    """
    torch.manual_seed(0)
    # 64-sample cap; 5 batches of 48 = 240 tokens >> 64, so Phase C of
    # the Vitter reservoir kicks in and the RNG seed matters.
    inputs = [torch.randn(48, 8, dtype=torch.bfloat16) for _ in range(5)]
    original_cap = s2pw._LAYER_INPUT_MAX_SAMPLES
    s2pw._LAYER_INPUT_MAX_SAMPLES = 64
    try:

        def _drive():
            s2pw._reset_state_for_tests()
            s2pw.setup(
                cov_storage_dtype="float16",
                layer_idx_to_rank={7: 0},
            )
            for x in inputs:
                s2pw._on_layer_in_callback(layer_idx=7, hidden=x)
            return s2pw._get_state().layer_input_reservoir[7].buffer.clone()

        buf_a = _drive()
        buf_b = _drive()
        assert torch.equal(buf_a, buf_b)
    finally:
        s2pw._LAYER_INPUT_MAX_SAMPLES = original_cap


# ---------------------------------------------------------------------------
# §7.b — round-trip: writer → sidecar → reader → REAM consumer.
# ---------------------------------------------------------------------------
def test_roundtrip_layer_input_reservoir_populated(tmp_path):
    """Writer-fed reservoir survives the bf16 sidecar round-trip + reader hydration."""
    s2pw._reset_state_for_tests()
    s2pw.setup(
        cov_storage_dtype="float16",
        n_experts=2,
        top_k=1,
        layer_idx_to_rank={0: 0, 1: 1},
    )

    # Feed asymmetric counts so the per-layer .seen differs.
    torch.manual_seed(42)
    s2pw._on_layer_in_callback(
        layer_idx=0, hidden=torch.randn(64, 16, dtype=torch.bfloat16),
    )
    s2pw._on_layer_in_callback(
        layer_idx=1, hidden=torch.randn(128, 16, dtype=torch.bfloat16),
    )
    # Tell the writer how many tokens each layer saw so the reader's
    # full-hit threshold passes (the partial-hit guard inspects
    # ``payload.total_tokens_per_layer`` against the per-layer max).
    s2pw.record_batch_token_count(layer_idx=0, n_tokens=1000)
    s2pw.record_batch_token_count(layer_idx=1, n_tokens=1000)

    jsonl = _jsonl(tmp_path)
    s2pw.dump_stage2_profile(jsonl)

    payload = load_stage2_profile_v3(jsonl)
    assert payload is not None
    assert isinstance(payload.layer_input_reservoir, list)
    assert len(payload.layer_input_reservoir) == 2
    assert payload.layer_input_reservoir[0].dtype == torch.bfloat16
    assert payload.layer_input_reservoir[0].shape == (64, 16)
    assert payload.layer_input_reservoir[1].shape == (128, 16)

    # Reader hydrates the live accumulator with the saved buffer.
    cov_acc = InputCovarianceAccumulator()
    cov_acc.set_storage_dtype(torch.float16)
    provider = Stage2ProfileCacheProvider(cov_acc=cov_acc)
    provider.payload = payload

    ctx = PipelineContext()
    ctx.set("_layer_rank", 0)
    ctx.set("layer_ref", SimpleNamespace(layer_idx=0, num_routed_experts=2))
    ctx.set("ream_acc", ReamCostAccumulator())
    layer_input_acc = _LayerInputAccumulator(max_samples=8192, seed=0)
    ctx.set("layer_input_acc", layer_input_acc)
    provider.on_layer_setup(ctx)

    assert ctx.has("stage2_profile_full_hit")
    assert layer_input_acc.buffer is not None
    assert layer_input_acc.buffer.shape == (64, 16)
    assert layer_input_acc.seen == 64
    assert torch.equal(
        layer_input_acc.buffer,
        payload.layer_input_reservoir[0],
    )


# ---------------------------------------------------------------------------
# §7.b extra — placeholder fallback when no layer_in batches arrive.
# ---------------------------------------------------------------------------
def test_dump_emits_empty_placeholder_when_no_layer_in_fired(tmp_path):
    """No hook calls → bf16 (0, 0) placeholders preserved (back-compat reader)."""
    s2pw._reset_state_for_tests()
    s2pw.setup(
        cov_storage_dtype="float16",
        n_experts=2,
        top_k=1,
        layer_idx_to_rank={0: 0, 1: 1},
    )
    # Touch ream_acc so a non-empty sidecar gets written.
    s2pw._on_router_callback(
        layer_idx=0,
        logits=torch.zeros((4, 2), dtype=torch.float32),
        batch_offset=0,
    )
    s2pw._on_router_callback(
        layer_idx=1,
        logits=torch.zeros((4, 2), dtype=torch.float32),
        batch_offset=0,
    )

    jsonl = _jsonl(tmp_path)
    s2pw.dump_stage2_profile(jsonl)

    payload = load_stage2_profile_v3(jsonl)
    for t in payload.layer_input_reservoir:
        assert t.dtype == torch.bfloat16
        assert t.numel() == 0


# ---------------------------------------------------------------------------
# §7.e — resume mid-capture: checkpoint preserves accumulator state.
# ---------------------------------------------------------------------------
def test_checkpoint_resume_preserves_accumulator_state(tmp_path):
    """Dump checkpoint with partial reservoirs → reload → resume byte-identical."""
    s2pw._reset_state_for_tests()
    s2pw.setup(
        cov_storage_dtype="float16",
        layer_idx_to_rank={0: 0, 1: 1, 2: 2, 3: 3},
    )

    torch.manual_seed(11)
    s2pw._on_layer_in_callback(
        layer_idx=0, hidden=torch.randn(32, 8, dtype=torch.bfloat16),
    )
    s2pw._on_layer_in_callback(
        layer_idx=2, hidden=torch.randn(48, 8, dtype=torch.bfloat16),
    )

    snapshot = {
        li: (acc.buffer.clone(), acc.seen, acc.max_samples)
        for li, acc in s2pw._get_state().layer_input_reservoir.items()
    }

    ckpt_path = tmp_path / "writer.ckpt"
    s2pw.dump_stage2_profile_checkpoint(ckpt_path)

    s2pw._reset_state_for_tests()
    s2pw.setup(
        cov_storage_dtype="float16",
        layer_idx_to_rank={0: 0, 1: 1, 2: 2, 3: 3},
    )
    s2pw.load_stage2_profile_checkpoint(ckpt_path)

    resumed = s2pw._get_state().layer_input_reservoir
    assert set(resumed) == set(snapshot)
    for li, (orig_buf, orig_seen, orig_max) in snapshot.items():
        acc = resumed[li]
        assert isinstance(acc, _LayerInputAccumulator)
        assert acc.seen == orig_seen
        assert acc.max_samples == orig_max
        assert torch.equal(acc.buffer, orig_buf)

    # Resume: feed more tokens and confirm the buffer keeps growing.
    extra = torch.randn(8, 8, dtype=torch.bfloat16)
    s2pw._on_layer_in_callback(layer_idx=0, hidden=extra)
    acc = s2pw._get_state().layer_input_reservoir[0]
    # Under max_samples (default 8192), the buffer concatenates.
    assert acc.buffer.shape[0] == snapshot[0][0].shape[0] + 8
    assert acc.seen == snapshot[0][1] + 8


# ---------------------------------------------------------------------------
# §7.e (MEDIUM-fix follow-up) — checkpoint resume is byte-identical EVEN
# after the per-layer reservoir entered Phase C (RNG-consuming branch).
# Pre-fix: `dump_stage2_profile_checkpoint` serialised only
# `(buffer, seen, max_samples)` and the loader re-initialised
# `_LayerInputAccumulator(seed=layer_idx)`. After Phase C consumed N draws,
# the resumed generator would re-emit the same draws (NOT continue from
# the consumed state). This test forces Phase C entry pre-checkpoint and
# asserts byte-equality against a no-resume reference.
# ---------------------------------------------------------------------------
def test_checkpoint_resume_byte_identical_after_phase_c(tmp_path):
    """Phase-C-active checkpoint resume MUST be byte-identical to no-resume.

    Setup:
      * Force Phase C with a tiny cap (max_samples=64).
      * The accumulator is constructed inside the writer via
        ``_on_layer_in_callback`` with the per-layer-idx seed.
      * We monkey-patch ``_LAYER_INPUT_MAX_SAMPLES`` to 64 so the writer's
        lazy constructor uses the small cap.

    Critical setup detail: ``_LayerInputAccumulator.add`` Phase A drops the
    over-cap tail of the FIRST call (profiling.py:46-55 docstring -- the
    deterministic-prefix contract that matches the scalar-loop baseline).
    To force Phase C entry the test must feed multiple small chunks so the
    accumulator runs Phase A on chunk 1, Phase B on chunk 2, and then Phase
    C from chunk 3 onward. We use 50-token chunks for clarity.

    Procedure:
      A. Reference path: feed 8 chunks of 50 tokens (= 400 total) straight;
         record final buffer + generator state.
      B. Resume path: feed 4 chunks (= 200), dump checkpoint, reset+load,
         feed remaining 4 chunks (= 200); record final buffer.
      C. Assert buffer A == buffer B byte-for-byte AND that the post-load
         generator state matches the reference's mid-stream state. With the
         pre-fix 3-tuple payload these diverge once Phase C consumes the
         per-layer RNG -- the loader re-emits seed=layer_idx so the resume
         half walks the same RNG prefix the reference's first half already
         consumed.
    """
    layer_idx = 0
    cap = 64
    n_chunks = 8
    chunk_size = 50  # > cap/2 so Phase C definitely runs once buffer fills.
    cut = n_chunks // 2

    torch.manual_seed(123)
    chunks = [
        torch.randn(chunk_size, 8, dtype=torch.bfloat16)
        for _ in range(n_chunks)
    ]

    saved_cap = s2pw._LAYER_INPUT_MAX_SAMPLES
    try:
        s2pw._LAYER_INPUT_MAX_SAMPLES = cap

        # --- Reference path: no resume, single process feeds all chunks ---
        s2pw._reset_state_for_tests()
        s2pw.setup(
            cov_storage_dtype="float16",
            layer_idx_to_rank={layer_idx: 0},
        )
        for chunk in chunks:
            s2pw._on_layer_in_callback(layer_idx=layer_idx, hidden=chunk)
        acc_ref = s2pw._get_state().layer_input_reservoir[layer_idx]
        assert acc_ref.buffer is not None
        assert acc_ref.seen == n_chunks * chunk_size
        assert acc_ref.buffer.shape[0] == cap
        reference_buffer = acc_ref.buffer.clone()
        reference_gen_state_final = acc_ref._generator.get_state().clone()

        # Also record what the gen state looks like at the cut point (the
        # state we expect ``load_checkpoint`` to restore -- this is the
        # invariant the bug breaks).
        s2pw._reset_state_for_tests()
        s2pw.setup(
            cov_storage_dtype="float16",
            layer_idx_to_rank={layer_idx: 0},
        )
        for chunk in chunks[:cut]:
            s2pw._on_layer_in_callback(layer_idx=layer_idx, hidden=chunk)
        gen_state_at_cut_ref = (
            s2pw._get_state().layer_input_reservoir[layer_idx]
            ._generator.get_state().clone()
        )
        # Sanity: by cut=4 chunks (200 tokens > cap=64), the generator
        # MUST have been advanced past its seed-only state.
        seed_only_state = (
            torch.Generator(device="cpu").manual_seed(layer_idx).get_state()
        )
        assert not torch.equal(gen_state_at_cut_ref, seed_only_state), (
            "Phase C did not consume the RNG by cut=4 chunks -- test "
            "setup is wrong, chunk sizing must force Phase C entry."
        )

        # --- Resume path: feed first half, dump, reset+load, feed rest ---
        s2pw._reset_state_for_tests()
        s2pw.setup(
            cov_storage_dtype="float16",
            layer_idx_to_rank={layer_idx: 0},
        )
        for chunk in chunks[:cut]:
            s2pw._on_layer_in_callback(layer_idx=layer_idx, hidden=chunk)
        acc_pre = s2pw._get_state().layer_input_reservoir[layer_idx]
        assert acc_pre.seen == cut * chunk_size
        assert acc_pre.buffer.shape[0] == cap

        ckpt_path = tmp_path / "writer_phase_c.ckpt"
        s2pw.dump_stage2_profile_checkpoint(ckpt_path)

        s2pw._reset_state_for_tests()
        s2pw.setup(
            cov_storage_dtype="float16",
            layer_idx_to_rank={layer_idx: 0},
        )
        s2pw.load_stage2_profile_checkpoint(ckpt_path)
        acc_loaded = s2pw._get_state().layer_input_reservoir[layer_idx]
        # The loaded generator state MUST match the reference's mid-stream
        # state. With the pre-fix 3-tuple loader this is a seed-only re-init
        # (NOT equal to the reference) so the assertion fails loudly here.
        assert torch.equal(
            acc_loaded._generator.get_state(), gen_state_at_cut_ref,
        ), (
            "Post-load generator state does not match the reference's "
            "mid-stream state. The 3-tuple payload re-seeded the generator "
            "instead of restoring its consumed-stream position."
        )

        # Continue feeding the second half.
        for chunk in chunks[cut:]:
            s2pw._on_layer_in_callback(layer_idx=layer_idx, hidden=chunk)
        acc_resume = s2pw._get_state().layer_input_reservoir[layer_idx]

        assert acc_resume.seen == n_chunks * chunk_size
        assert acc_resume.buffer.shape == reference_buffer.shape
        assert torch.equal(acc_resume.buffer, reference_buffer), (
            "Resumed Phase-C buffer diverges from reference -- generator "
            "state was not restored byte-identically across the dump/load "
            "cycle."
        )
        assert torch.equal(
            acc_resume._generator.get_state(), reference_gen_state_final,
        ), (
            "Resumed Phase-C generator state diverges from reference -- "
            "future draws will not match."
        )
    finally:
        s2pw._LAYER_INPUT_MAX_SAMPLES = saved_cap


# ---------------------------------------------------------------------------
# §7.g — hidden bug: full-hit + empty-reservoir + cost_alignment=output
#         → reader demotes to partial hit.
# ---------------------------------------------------------------------------
def _build_payload_with_empty_reservoir(
    n_layers: int = 2,
) -> Stage2ProfilePayloadV3:
    """Build a sidecar that mimics a pre-CRITICAL-1 capture (empty reservoirs)."""
    gate_logit_profiles = {
        lr: [(0, torch.ones((1000, 2), dtype=torch.float32))]
        for lr in range(n_layers)
    }
    cov_acc = {
        (lr, e, m): torch.eye(4, dtype=torch.float16)
        for lr in range(n_layers)
        for e in range(2)
        for m in ("gate_proj", "down_proj")
    }
    return Stage2ProfilePayloadV3(
        format_version=3,
        schema_version=SCHEMA_VERSIONS["stage2_profile"],
        model_hash="h",
        n_layers=n_layers,
        n_experts=2,
        top_k=1,
        cov_storage_dtype="float16",
        total_tokens_per_layer=torch.full((n_layers,), 1000, dtype=torch.int64),
        gate_logit_profiles=gate_logit_profiles,
        sim_tensor=torch.zeros((n_layers, 2, 2), dtype=torch.float64),
        neuron_act_sum={
            (lr, e): torch.zeros((4,), dtype=torch.float32)
            for lr in range(n_layers) for e in range(2)
        },
        neuron_act_count={
            (lr, e): 5 for lr in range(n_layers) for e in range(2)
        },
        cov_acc=cov_acc,
        cov_token_count={k: 5 for k in cov_acc},
        layer_input_reservoir=[
            torch.zeros((0, 0), dtype=torch.bfloat16) for _ in range(n_layers)
        ],
    )


def test_demote_empty_reservoir_output_alignment():
    """Plan §5.b: cost_alignment='output' + empty reservoir → partial hit."""
    payload = _build_payload_with_empty_reservoir()
    cov_acc = InputCovarianceAccumulator()
    cov_acc.set_storage_dtype(torch.float16)

    provider = Stage2ProfileCacheProvider(
        cov_acc=cov_acc,
        cost_alignment="output",
    )
    provider.payload = payload

    ctx = PipelineContext()
    ctx.set("_layer_rank", 0)
    ctx.set("layer_ref", SimpleNamespace(layer_idx=0, num_routed_experts=2))
    ctx.set("ream_acc", ReamCostAccumulator())
    layer_input_acc = _LayerInputAccumulator(max_samples=128, seed=0)
    ctx.set("layer_input_acc", layer_input_acc)

    provider.on_layer_setup(ctx)

    # Demoted to partial; full-hit NOT set.
    assert ctx.has("stage2_profile_partial_hit")
    assert ctx.get("stage2_profile_partial_hit") is True
    assert not ctx.has("stage2_profile_full_hit")
    # Live forward will populate layer_input_acc; reader left it untouched.
    assert layer_input_acc.buffer is None


def test_no_demote_when_cost_alignment_is_pre():
    """cost_alignment='pre' + empty reservoir → still a full hit (legacy behaviour)."""
    payload = _build_payload_with_empty_reservoir()
    cov_acc = InputCovarianceAccumulator()
    cov_acc.set_storage_dtype(torch.float16)

    provider = Stage2ProfileCacheProvider(
        cov_acc=cov_acc,
        cost_alignment="pre",
    )
    provider.payload = payload

    ctx = PipelineContext()
    ctx.set("_layer_rank", 0)
    ctx.set("layer_ref", SimpleNamespace(layer_idx=0, num_routed_experts=2))
    ctx.set("ream_acc", ReamCostAccumulator())
    layer_input_acc = _LayerInputAccumulator(max_samples=128, seed=0)
    ctx.set("layer_input_acc", layer_input_acc)

    provider.on_layer_setup(ctx)

    # Output-space cost was never going to read the reservoir → no demote.
    assert ctx.has("stage2_profile_full_hit")
    assert not ctx.has("stage2_profile_partial_hit")


def test_full_hit_when_reservoir_populated_under_output_alignment():
    """cost_alignment='output' + non-empty reservoir → full hit + hydration."""
    payload = _build_payload_with_empty_reservoir()
    # Overwrite the rank-0 reservoir with real data.
    payload.layer_input_reservoir[0] = torch.arange(
        16 * 4, dtype=torch.float32,
    ).reshape(16, 4).to(torch.bfloat16)

    cov_acc = InputCovarianceAccumulator()
    cov_acc.set_storage_dtype(torch.float16)
    provider = Stage2ProfileCacheProvider(
        cov_acc=cov_acc,
        cost_alignment="output",
    )
    provider.payload = payload

    ctx = PipelineContext()
    ctx.set("_layer_rank", 0)
    ctx.set("layer_ref", SimpleNamespace(layer_idx=0, num_routed_experts=2))
    ctx.set("ream_acc", ReamCostAccumulator())
    layer_input_acc = _LayerInputAccumulator(max_samples=128, seed=0)
    ctx.set("layer_input_acc", layer_input_acc)

    provider.on_layer_setup(ctx)

    assert ctx.has("stage2_profile_full_hit")
    assert not ctx.has("stage2_profile_partial_hit")
    assert layer_input_acc.buffer is not None
    assert layer_input_acc.buffer.shape == (16, 4)
    assert layer_input_acc.seen == 16


# ---------------------------------------------------------------------------
# Pattern N — patch copy stays in sync with the canonical callback list.
# ---------------------------------------------------------------------------
def test_layer_in_hook_present_in_vllm_patch_copy():
    """The patch-shipped vllm/calibration_stage2_profile.py mirrors the layer_in hook."""
    patch_path = (
        Path(__file__).resolve().parents[1]
        / "patches" / "vllm_calibration_stage2_profile.patch"
    )
    body = patch_path.read_text(encoding="utf-8")
    assert "_layer_in_handler" in body
    assert "register_callback(\"layer_in\", _layer_in_handler)" in body


def test_layer_in_hook_name_present_in_vllm_hooks_patch():
    """vllm_calibration_hooks.patch declares the new hook name + env gate."""
    patch_path = (
        Path(__file__).resolve().parents[1]
        / "patches" / "vllm_calibration_hooks.patch"
    )
    body = patch_path.read_text(encoding="utf-8")
    assert "\"layer_in\"" in body
    assert "_CAPTURE_LAYER_IN" in body
    assert "VLLM_CALIB_CAPTURE_LAYER_IN" in body
