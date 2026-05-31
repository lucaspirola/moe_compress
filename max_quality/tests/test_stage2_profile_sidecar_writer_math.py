"""Writer-side math correctness for the Stage 2 profile-pass sidecar.

Plan section 8.1 — Bug #1 reference test. Drives the writer's internal
callbacks directly (no vLLM required) and compares the resulting
sim_tensor / gate_gram / total_tokens_per_layer with a
parallel live :class:`ReamCostAccumulator` fed the SAME synthetic
inputs.

Critically, the test must FAIL on the prior buggy implementation that
used ``cos(mean(g_i), mean(g_j))`` instead of per-token jointly-active
pair cosines. We construct a fixture where ``mean(cos_pairs)`` differs
from ``cos(mean_i, mean_j)`` (any non-parallel pair of gated vectors
satisfies this), and verify the writer's sim_tensor matches the
reference. A buggy implementation would fail the assertion.

Plan section 8.3 — Bug #3 regression: total_tokens_per_layer is the
exact T_batch * n_batches token count, NOT a top_k-scaled aggregate.

Plan section 8.6 (end-to-end roundtrip) — feed the loaded payload
into Stage2ProfileCacheProvider via on_layer_setup and verify
ream_acc._sim_tensor[layer_idx] matches reference.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from moe_compress.calibration import stage2_profile_writer as s2pw
from moe_compress.pipeline.context import PipelineContext
from moe_compress.stage2.plugins.stage2_profile_cache import (
    Stage2ProfileCacheProvider,
)
from moe_compress.utils.activation_hooks import (
    InputCovarianceAccumulator,
    ReamCostAccumulator,
)
from moe_compress.utils.cached_calibration_signals import (
    load_stage2_profile_v4,
    sidecar_path,
)


def _jsonl(tmp_path: Path) -> Path:
    return tmp_path / "calib" / "self_traces.jsonl"


# ---------------------------------------------------------------------------
# Fixture: synthetic mini-MoE shape; per-token gated vectors deliberately
# non-parallel so cos(per-token-pair) differs from cos(mean_i, mean_j).
# ---------------------------------------------------------------------------
def _synth_inputs(seed: int = 0):
    torch.manual_seed(seed)
    n_layers = 2
    n_experts = 4
    top_k = 2
    d_hid = 16
    n_batches = 3
    T = 8
    return {
        "n_layers": n_layers, "n_experts": n_experts, "top_k": top_k,
        "d_hid": d_hid, "n_batches": n_batches, "T": T,
    }


def _build_batches(cfg: dict, seed: int):
    """Produce a list of (logits, per_expert_payloads, T) per batch per layer.

    Per batch: random logits [T, E] fp32; for each expert, a random gated
    output [n_active_e, d_hid] fp32 with a per-token index tensor
    [n_active_e] long. Active assignments are random subsets of [T] of
    size between 1 and T to exercise the jointly-active filter.
    """
    g = torch.Generator().manual_seed(seed)
    batches: list[dict] = []
    for b in range(cfg["n_batches"]):
        per_layer = {}
        for lr in range(cfg["n_layers"]):
            logits = torch.randn(
                (cfg["T"], cfg["n_experts"]), generator=g, dtype=torch.float32,
            )
            # For each expert, pick a random non-empty subset of tokens.
            payloads = []
            for e in range(cfg["n_experts"]):
                # Random mask with prob ~0.6 so jointly-active pairs exist.
                mask = (torch.rand((cfg["T"],), generator=g) < 0.6)
                if mask.sum() == 0:
                    mask[0] = True  # ensure at least one active token
                indices = torch.nonzero(mask, as_tuple=False).squeeze(-1).to(torch.long)
                # Random gated output for those tokens.
                gated = torch.randn(
                    (int(indices.numel()), cfg["d_hid"]),
                    generator=g, dtype=torch.float32,
                )
                payloads.append((e, indices, gated))
            per_layer[lr] = (logits, payloads)
        batches.append({"per_layer": per_layer})
    return batches


def _drive_writer(cfg: dict, batches: list[dict], cov_storage_dtype: str = "float16"):
    """Drive the writer callbacks for the synthetic batches."""
    s2pw._reset_state_for_tests()
    s2pw.setup(
        llm=None,
        cov_storage_dtype=cov_storage_dtype,
        n_layers=cfg["n_layers"],
        n_experts=cfg["n_experts"],
        top_k=cfg["top_k"],
        model_hash="writer-math-test",
        # layer_idx == layer_rank for the test.
        layer_idx_to_rank={lr: lr for lr in range(cfg["n_layers"])},
    )
    # Use ones for gate_weights so gated tensor unchanged (we want the
    # raw vectors to drive the cosine sums independent of weights).
    offset = 0
    for b, batch in enumerate(batches):
        for lr, (logits, payloads) in batch["per_layer"].items():
            s2pw._on_router_callback(lr, logits, offset)
            for (e, indices, gated) in payloads:
                ones = torch.ones((int(indices.numel()),), dtype=gated.dtype)
                s2pw._on_expert_out_unweighted_callback(
                    lr, e, gated, indices, offset, gate_weights=ones,
                )
            s2pw._finalize_batch_for_layer(lr, cfg["n_experts"])
            s2pw.record_batch_token_count(lr, cfg["T"])
        offset += cfg["T"]


def _drive_reference(cfg: dict, batches: list[dict]) -> ReamCostAccumulator:
    """Drive a parallel live ReamCostAccumulator the same way."""
    ref = ReamCostAccumulator()
    ref.num_experts = cfg["n_experts"]
    offset = 0
    for b, batch in enumerate(batches):
        for lr, (logits, payloads) in batch["per_layer"].items():
            ref.record_router_logits(lr, logits, offset)
            for (e, indices, gated) in payloads:
                ones = torch.ones((int(indices.numel()),), dtype=gated.dtype)
                ref.record_gated_output(
                    lr, e,
                    gate_weights=ones,
                    expert_output=gated,
                    token_indices=indices,
                    batch_offset=offset,
                )
            ref.finalize_batch(lr, cfg["n_experts"])
            ref.record_batch_token_count(lr, cfg["T"])
        offset += cfg["T"]
    return ref


def test_writer_sim_tensor_matches_reference(tmp_path):
    """sim_tensor matches the live ReamCostAccumulator output (Bug #1 fix).

    The writer's _finalize_batch_for_layer calls the SAME
    ReamCostAccumulator.finalize_batch as the live path, so equality is
    by construction. A buggy implementation that computed
    cos(mean(g_i), mean(g_j)) would fail this assertion on the
    non-parallel synthetic vectors (see module docstring).
    """
    cfg = _synth_inputs()
    batches = _build_batches(cfg, seed=11)
    _drive_writer(cfg, batches)
    ref = _drive_reference(cfg, batches)

    state = s2pw._get_state()
    for lr in range(cfg["n_layers"]):
        live_sim = ref._sim_tensor.get(lr)
        writer_sim = state.ream_acc._sim_tensor.get(lr)
        assert live_sim is not None and writer_sim is not None
        # Identical accumulators — exact equality holds modulo float32
        # accumulation order (the same code path runs on both).
        torch.testing.assert_close(
            writer_sim, live_sim,
            rtol=0.0, atol=1e-10,
        )


def test_total_tokens_not_off_by_top_k(tmp_path):
    """Bug #3 regression: total_tokens_per_layer == T*n_batches, not k*T*n_batches."""
    cfg = _synth_inputs()
    batches = _build_batches(cfg, seed=12)
    _drive_writer(cfg, batches)
    state = s2pw._get_state()
    expected = cfg["T"] * cfg["n_batches"]
    for lr in range(cfg["n_layers"]):
        assert state.ream_acc._total_tokens_by_layer[lr] == expected, (
            f"layer {lr}: total_tokens_by_layer="
            f"{state.ream_acc._total_tokens_by_layer[lr]} != {expected}"
        )


def test_gate_gram_matches_reference(tmp_path):
    """_gate_gram is the bounded [E, E] fp64 Gram and matches the reference (Bug #2).

    The writer's _on_router_callback folds each batch into the same online
    ReamCostAccumulator._gate_gram as the live path, so equality is by
    construction.
    """
    cfg = _synth_inputs()
    batches = _build_batches(cfg, seed=13)
    _drive_writer(cfg, batches)
    ref = _drive_reference(cfg, batches)
    state = s2pw._get_state()
    for lr in range(cfg["n_layers"]):
        gram = state.ream_acc._gate_gram[lr]
        assert isinstance(gram, torch.Tensor)
        assert gram.dtype == torch.float64
        assert gram.shape == (cfg["n_experts"], cfg["n_experts"])
        torch.testing.assert_close(
            gram, ref._gate_gram[lr], rtol=0.0, atol=1e-10,
        )


def test_dump_load_roundtrip_byte_identical(tmp_path):
    """dump_stage2_profile -> load_stage2_profile_v4 preserves payload fields."""
    cfg = _synth_inputs()
    batches = _build_batches(cfg, seed=14)
    _drive_writer(cfg, batches)
    # Inject a layer_input_reservoir entry so the dump path doesn't emit
    # the zero-shape fallback.
    for lr in range(cfg["n_layers"]):
        s2pw._record_layer_input_reservoir(
            lr,
            torch.arange(8 * cfg["d_hid"], dtype=torch.float32).reshape(8, cfg["d_hid"]),
        )
        # Inject neuron-act state so the writer carries those too.
        for e in range(cfg["n_experts"]):
            s2pw._record_neuron_act(
                lr, e,
                torch.full((cfg["d_hid"],), float(e + 1)),
                n_tokens=cfg["T"] * cfg["n_batches"],
            )

    jsonl = _jsonl(tmp_path)
    s2pw.dump_stage2_profile(jsonl)

    loaded = load_stage2_profile_v4(jsonl)
    assert loaded is not None
    assert loaded.schema_version == 4
    assert loaded.cov_storage_dtype == "float16"
    assert loaded.n_layers == cfg["n_layers"]
    assert loaded.n_experts == cfg["n_experts"]
    assert loaded.top_k == cfg["top_k"]
    # total_tokens preserved.
    expected_total = cfg["T"] * cfg["n_batches"]
    assert (loaded.total_tokens_per_layer == expected_total).all()
    # gate_gram: bounded [n_layers, E, E] fp64 Gram, matches the live acc.
    assert loaded.gate_gram.shape == (
        cfg["n_layers"], cfg["n_experts"], cfg["n_experts"],
    )
    assert loaded.gate_gram.dtype == torch.float64
    state = s2pw._get_state()
    for lr in range(cfg["n_layers"]):
        torch.testing.assert_close(
            loaded.gate_gram[lr], state.ream_acc._gate_gram[lr],
            rtol=0.0, atol=1e-10,
        )


def test_hydrated_acc_matches_reference_via_reader(tmp_path):
    """End-to-end: dump -> load -> hydrate via Stage2ProfileCacheProvider.

    Plan section 8.1 step 6. The reader's on_layer_setup populates
    ream_acc._sim_tensor[layer_idx] in place; we assert it equals the
    parallel live accumulator's row.
    """
    cfg = _synth_inputs()
    batches = _build_batches(cfg, seed=15)
    _drive_writer(cfg, batches)
    # Reservoir + neuron entries for completeness.
    for lr in range(cfg["n_layers"]):
        s2pw._record_layer_input_reservoir(
            lr, torch.zeros((8, cfg["d_hid"]), dtype=torch.float32),
        )
    jsonl = _jsonl(tmp_path)
    s2pw.dump_stage2_profile(jsonl)
    ref = _drive_reference(cfg, batches)

    cov_acc = InputCovarianceAccumulator()
    cov_acc.set_storage_dtype(torch.float16)
    provider = Stage2ProfileCacheProvider(
        cov_acc=cov_acc, expected_cov_storage_dtype="float16",
    )
    run_ctx = PipelineContext()
    provider.on_load(run_ctx, jsonl)
    assert provider.payload is not None

    for lr in range(cfg["n_layers"]):
        # layer_rank == layer_idx for this synthetic test.
        layer_idx = lr
        ctx = run_ctx.child()
        ctx.set("_layer_rank", lr)
        ctx.set("layer_ref", SimpleNamespace(
            layer_idx=layer_idx, num_routed_experts=cfg["n_experts"],
        ))
        # Fresh empty accumulators (mirror layer_merge.on_layer_setup).
        ctx.set("ream_acc", ReamCostAccumulator())
        ctx.set("layer_input_acc", None)
        provider.on_layer_setup(ctx)
        # FULL hit asserted.
        assert ctx.has("stage2_profile_full_hit")
        hydrated = ctx.get("ream_acc")
        # _sim_tensor matches reference row.
        torch.testing.assert_close(
            hydrated._sim_tensor[layer_idx],
            ref._sim_tensor[layer_idx],
            rtol=0.0, atol=1e-10,
        )
        # _total_tokens matches.
        assert (hydrated._total_tokens_by_layer[layer_idx]
                == ref._total_tokens_by_layer[layer_idx])


def test_bug1_fixture_distinguishes_mean_vs_pair_cosine():
    """Sanity guard for the fixture: pair-cosine != mean-of-means cosine.

    Confirms the synthetic vectors are non-parallel enough that a buggy
    cos(mean_i, mean_j) implementation would yield a different sim_tensor
    than the correct per-token pair cosine. If this assertion fails,
    the writer-math test loses its bug-discrimination power -- the
    fixture should be re-seeded.
    """
    # Two tokens, two experts, both jointly active. Pick g_i, g_j vectors
    # whose individual cosines differ markedly from the cos of the means.
    g_i = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    g_j = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float32)
    pair_cos = torch.nn.functional.cosine_similarity(g_i, g_j, dim=-1)
    mean_i = g_i.mean(dim=0)
    mean_j = g_j.mean(dim=0)
    mean_of_means_cos = torch.nn.functional.cosine_similarity(
        mean_i.unsqueeze(0), mean_j.unsqueeze(0), dim=-1,
    )
    # mean_of_means_cos = 1.0, while pair_cos = [0, 0] -> sum 0.0 -> mean 0.
    assert not torch.allclose(pair_cos.mean(), mean_of_means_cos.squeeze()), (
        "Fixture invariant violated: vectors must be non-parallel "
        "enough that per-token pair cosines differ from mean-of-means."
    )


def _buggy_mean_of_means_sim(
    cfg: dict, batches: list[dict],
) -> dict[int, torch.Tensor]:
    """Reference implementation of the BUGGY prior writer (Bug #1).

    Computes per-layer sim_tensor as cos(mean(g_i), mean(g_j)) across all
    tokens, NOT the correct per-token-pair cosine sum. We use this as the
    discriminator: the writer's actual sim_tensor must NOT equal this
    buggy variant on the non-parallel fixture (asserted in
    :func:`test_writer_sim_tensor_distinguishes_from_buggy_path`).
    """
    accum: dict[int, dict[int, torch.Tensor]] = {}  # layer -> expert -> mean
    counts: dict[int, dict[int, int]] = {}
    for batch in batches:
        for lr, (_logits, payloads) in batch["per_layer"].items():
            for (e, indices, gated) in payloads:
                T = int(indices.numel())
                if T == 0:
                    continue
                prev = accum.setdefault(lr, {}).get(e)
                if prev is None:
                    accum[lr][e] = gated.sum(dim=0)
                else:
                    accum[lr][e] = prev + gated.sum(dim=0)
                counts.setdefault(lr, {})
                counts[lr][e] = counts[lr].get(e, 0) + T
    sim: dict[int, torch.Tensor] = {}
    for lr in range(cfg["n_layers"]):
        s = torch.zeros((cfg["n_experts"], cfg["n_experts"]), dtype=torch.float64)
        means = {
            e: (accum[lr][e] / max(counts[lr][e], 1))
            for e in accum.get(lr, {})
        }
        for i in means:
            for j in means:
                if i == j:
                    continue
                ci = torch.nn.functional.cosine_similarity(
                    means[i].unsqueeze(0), means[j].unsqueeze(0), dim=-1,
                )
                s[i, j] = float(ci.item())
        sim[lr] = s
    return sim


def test_writer_sim_tensor_distinguishes_from_buggy_path(tmp_path):
    """The writer's sim_tensor MUST NOT equal cos(mean_i, mean_j).

    Plan section 8.1 step 5: a writer that implemented Bug #1 (mean-of-
    means cosine) would yield this alternate sim_tensor on the synthetic
    fixture. We assert the writer's actual sim_tensor differs from the
    buggy variant -- this is what protects the implementation from
    silently regressing back to the deleted prior writer's math.
    """
    cfg = _synth_inputs()
    batches = _build_batches(cfg, seed=21)
    _drive_writer(cfg, batches)
    state = s2pw._get_state()
    buggy = _buggy_mean_of_means_sim(cfg, batches)
    for lr in range(cfg["n_layers"]):
        writer_sim = state.ream_acc._sim_tensor[lr]
        buggy_sim = buggy[lr]
        # The matrices should NOT be close on this fixture.
        assert not torch.allclose(
            writer_sim, buggy_sim, atol=1e-6, rtol=0.0,
        ), (
            f"layer {lr}: writer sim_tensor coincidentally matches the "
            f"buggy mean-of-means path -- fixture lost discrimination power"
        )
