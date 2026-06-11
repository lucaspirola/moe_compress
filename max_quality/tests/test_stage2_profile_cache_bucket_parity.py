"""C10 parity test: pre-bucketed hydration == old 4-scan hydration.

The C10 speed-up replaces the four full ``payload`` dict scans in
:meth:`Stage2ProfileCacheProvider.on_layer_setup` (one each for
``neuron_act_sum`` / ``neuron_act_count`` / ``cov_acc`` /
``cov_token_count``, each filtering ``int(lr) == int(layer_rank)``) with a
per-``layer_rank`` index built ONCE and cached on the instance.

This test pins byte-identity: a multi-rank synthetic payload is hydrated
through BOTH

  * the OLD 4-scan path (reconstructed verbatim in ``_old_hydrate`` below —
    a faithful copy of the pre-C10 loop bodies), and
  * the NEW pre-bucketed path (the real ``provider.on_layer_setup``),

and every hydrated accumulator entry is compared with ``torch.equal`` (for
tensors) / ``==`` (for scalar ints). The two MUST be byte-identical for
every ``(layer_idx, e[, m])`` key, on every layer_rank.

Merge-only / opt-in (``profile_sidecar.enabled`` default false) — not
production — but landed byte-identically per the C10 HARD BAR.
"""
from __future__ import annotations

from types import SimpleNamespace

import torch

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
    Stage2ProfilePayloadV4,
)

_MODULES = ("gate_proj", "down_proj")


def _build_multirank_payload(
    *, n_layers: int = 3, n_experts: int = 4, hidden: int = 6,
) -> Stage2ProfilePayloadV4:
    """Synthetic payload spanning ``n_layers`` distinct layer_ranks.

    Tensors carry rank/expert-distinct values so a key-misattribution bug
    (wrong rank's row landing on a layer) would fail ``torch.equal``.
    Insertion order is intentionally NOT grouped by rank (interleaved
    across ranks) to stress the bucket's order-preservation contract.
    """
    torch.manual_seed(7)
    neuron_act_sum: dict[tuple[int, int], torch.Tensor] = {}
    neuron_act_count: dict[tuple[int, int], int] = {}
    cov_acc: dict[tuple[int, int, str], torch.Tensor] = {}
    cov_token_count: dict[tuple[int, int, str], int] = {}

    # Interleave experts in the outer loop so successive dict entries hop
    # between ranks (exercises the "preserve .items() insertion order"
    # requirement, not merely "group-by-rank happens to be sorted").
    for e in range(n_experts):
        for lr in range(n_layers):
            neuron_act_sum[(lr, e)] = torch.randn(hidden, dtype=torch.float32)
            neuron_act_count[(lr, e)] = 100 * lr + e + 1
            for m in _MODULES:
                cov_acc[(lr, e, m)] = torch.randn(
                    hidden, hidden, dtype=torch.float32,
                )
                cov_token_count[(lr, e, m)] = 1000 * lr + 10 * e + len(m)

    return Stage2ProfilePayloadV4(
        format_version=4,
        schema_version=SCHEMA_VERSIONS["stage2_profile"],
        model_hash="c10-parity",
        n_layers=n_layers,
        n_experts=n_experts,
        top_k=2,
        cov_storage_dtype="float16",
        total_tokens_per_layer=torch.full(
            (n_layers,), 1000, dtype=torch.int64,
        ),
        gate_gram=torch.randn(
            (n_layers, n_experts, n_experts), dtype=torch.float64,
        ),
        sim_tensor=torch.randn(
            (n_layers, n_experts, n_experts), dtype=torch.float64,
        ),
        neuron_act_sum=neuron_act_sum,
        neuron_act_count=neuron_act_count,
        cov_acc=cov_acc,
        cov_token_count=cov_token_count,
        layer_input_reservoir=[
            torch.randn((8, hidden), dtype=torch.bfloat16)
            for _ in range(n_layers)
        ],
    )


def _old_hydrate(payload, layer_rank, layer_idx, ream_acc, cov_acc):
    """Verbatim pre-C10 4-scan hydration of the four payload dicts.

    Copied from the pre-C10 ``on_layer_setup`` body (the loops at the old
    L268 / L271 / L283 / L288) so the test captures the original behaviour
    independent of the refactor under test.
    """
    live_dtype = cov_acc.storage_dtype
    for (lr, e), v in payload.neuron_act_sum.items():
        if int(lr) == int(layer_rank):
            ream_acc._neuron_act_sum[(layer_idx, int(e))] = v.clone()
    for (lr, e), c in payload.neuron_act_count.items():
        if int(lr) == int(layer_rank):
            ream_acc._neuron_act_count[(layer_idx, int(e))] = int(c)
    for (lr, e, m), cov_t in payload.cov_acc.items():
        if int(lr) == int(layer_rank):
            cov_acc.covariance[(layer_idx, int(e), str(m))] = cov_t.to(
                live_dtype,
            )
    for (lr, e, m), n in payload.cov_token_count.items():
        if int(lr) == int(layer_rank):
            cov_acc.token_count[(layer_idx, int(e), str(m))] = int(n)


# Map each layer_rank to a distinct layer_idx to exercise the rank→idx
# translation (idx != rank so a bug copying rank-keys would surface).
_RANK_TO_IDX = {0: 5, 1: 9, 2: 2}


def test_bucketed_hydration_byte_identical_to_old_4scan():
    payload = _build_multirank_payload()

    for layer_rank, layer_idx in _RANK_TO_IDX.items():
        # --- OLD path -------------------------------------------------------
        old_cov = InputCovarianceAccumulator()
        old_cov.set_storage_dtype(torch.float16)
        old_ream = ReamCostAccumulator()
        _old_hydrate(payload, layer_rank, layer_idx, old_ream, old_cov)

        # --- NEW path (real provider) --------------------------------------
        new_cov = InputCovarianceAccumulator()
        new_cov.set_storage_dtype(torch.float16)
        provider = Stage2ProfileCacheProvider(cov_acc=new_cov)
        provider.payload = payload
        ctx = PipelineContext()
        ctx.set("_layer_rank", layer_rank)
        ctx.set(
            "layer_ref",
            SimpleNamespace(layer_idx=layer_idx, num_routed_experts=4),
        )
        new_ream = ReamCostAccumulator()
        ctx.set("ream_acc", new_ream)
        ctx.set(
            "layer_input_acc",
            _LayerInputAccumulator(max_samples=8192, seed=0),
        )
        provider.on_layer_setup(ctx)
        assert ctx.has("stage2_profile_full_hit")

        # --- Parity: every hydrated entry byte-identical -------------------
        # neuron_act_sum (tensors)
        assert set(new_ream._neuron_act_sum) == set(old_ream._neuron_act_sum)
        for k, v in old_ream._neuron_act_sum.items():
            assert torch.equal(new_ream._neuron_act_sum[k], v), (
                f"neuron_act_sum mismatch at {k} (rank={layer_rank})"
            )
        # neuron_act_count (ints)
        assert new_ream._neuron_act_count == old_ream._neuron_act_count
        # cov_acc.covariance (cast tensors)
        assert set(new_cov.covariance) == set(old_cov.covariance)
        for k, v in old_cov.covariance.items():
            nv = new_cov.covariance[k]
            assert nv.dtype == v.dtype
            assert torch.equal(nv, v), (
                f"covariance mismatch at {k} (rank={layer_rank})"
            )
        # cov_acc.token_count (ints)
        assert new_cov.token_count == old_cov.token_count


def test_bucket_built_once_and_reused_across_layers():
    """The per-rank index is built lazily ONCE and cached on the instance."""
    payload = _build_multirank_payload()
    cov_acc = InputCovarianceAccumulator()
    cov_acc.set_storage_dtype(torch.float16)
    provider = Stage2ProfileCacheProvider(cov_acc=cov_acc)
    provider.payload = payload

    assert provider._rank_buckets is None  # not built until first layer

    def _run(layer_rank, layer_idx):
        ctx = PipelineContext()
        ctx.set("_layer_rank", layer_rank)
        ctx.set(
            "layer_ref",
            SimpleNamespace(layer_idx=layer_idx, num_routed_experts=4),
        )
        ctx.set("ream_acc", ReamCostAccumulator())
        ctx.set(
            "layer_input_acc",
            _LayerInputAccumulator(max_samples=8192, seed=0),
        )
        provider.on_layer_setup(ctx)

    _run(0, 5)
    built = provider._rank_buckets
    assert built is not None
    assert set(built) == {0, 1, 2}
    _run(1, 9)
    # Same cached object — not rebuilt.
    assert provider._rank_buckets is built


def test_bucket_holds_payload_references_not_copies():
    """Bucketed tensors are the SAME objects as the payload (no copy)."""
    payload = _build_multirank_payload()
    cov_acc = InputCovarianceAccumulator()
    cov_acc.set_storage_dtype(torch.float16)
    provider = Stage2ProfileCacheProvider(cov_acc=cov_acc)
    provider.payload = payload

    bucket = provider._rank_bucket_for(1)
    for e, v in bucket.neuron_act_sum:
        assert v is payload.neuron_act_sum[(1, e)]
    for e, m, cov_t in bucket.cov_acc:
        assert cov_t is payload.cov_acc[(1, e, m)]
