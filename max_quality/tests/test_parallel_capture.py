"""CPU tests for the data-parallel calibration-capture machinery.

No GPU, no vllm import, no monkeypatch. Covers:
  * shard_split: disjoint + complete partition of a synthetic JSONL.
  * merge_sidecars: merged reduction EXACTLY reproduces a single full run for
    reap_scores (count-weighted mean), per_expert_max (element-wise max),
    and routing_stats (freq-weighted mean), via the real dataclasses +
    save_*/load_* round-trip.

See ``tasks/CALIB_PARALLEL_CAPTURE_DESIGN.md``.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))
_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS))

from moe_compress.utils import merge_sidecars as ms  # noqa: E402
from moe_compress.utils.cached_calibration_signals import (  # noqa: E402
    SCHEMA_VERSIONS,
    BlockHiddenPayload,
    RoutingStatsPayload,
    Stage1PerExpertMaxPayload,
    Stage2ReapPayload,
    load_block_hidden,
    load_per_expert_max,
    load_reap_scores,
    load_routing_stats,
    save_block_hidden,
    save_per_expert_max,
    save_reap_scores,
    save_routing_stats,
)


def _load_shard_split():
    spec = importlib.util.spec_from_file_location(
        "shard_split", _SCRIPTS / "shard_split.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


shard_split = _load_shard_split()


# ===========================================================================
# shard_split tests
# ===========================================================================
def _write_jsonl(path: Path, n_rows: int, *, key: str = "_attempt_idx"):
    with path.open("w", encoding="utf-8") as f:
        for i in range(n_rows):
            row = {
                key: i,
                "messages": [
                    {"role": "user", "content": f"q{i}"},
                    {"role": "assistant", "content": f"a{i}"},
                ],
            }
            f.write(json.dumps(row) + "\n")


def test_shard_split_disjoint_complete(tmp_path):
    src = tmp_path / "self_traces.jsonl"
    _write_jsonl(src, 10)
    out_dir = tmp_path / "parallel"

    counts = shard_split.split_jsonl(src, 4, out_dir)

    # 10 rows / 4 shards, ceil(10/4)=3 -> [3, 3, 3, 1].
    assert counts == [3, 3, 3, 1]
    assert sum(counts) == 10

    # Collect keys from every shard file; assert disjoint + complete.
    all_keys: list[int] = []
    per_shard_sets = []
    for k in range(4):
        shard_file = out_dir / f"shard_{k}" / f"shard_{k}.jsonl"
        assert shard_file.is_file()
        keys = set()
        with shard_file.open() as f:
            for line in f:
                keys.add(json.loads(line)["_attempt_idx"])
        per_shard_sets.append(keys)
        all_keys.extend(keys)

    # Disjoint: no key in two shards.
    for i in range(4):
        for j in range(i + 1, 4):
            assert per_shard_sets[i].isdisjoint(per_shard_sets[j]), (
                f"shards {i} and {j} overlap: "
                f"{per_shard_sets[i] & per_shard_sets[j]}"
            )
    # Complete: union == full set, no dupes.
    assert sorted(all_keys) == list(range(10))


def test_shard_split_n_equals_rows(tmp_path):
    src = tmp_path / "t.jsonl"
    _write_jsonl(src, 4)
    counts = shard_split.split_jsonl(src, 4, tmp_path / "o")
    assert counts == [1, 1, 1, 1]


def test_shard_split_more_shards_than_rows_errors(tmp_path):
    src = tmp_path / "t.jsonl"
    _write_jsonl(src, 3)
    with pytest.raises(ValueError, match="exceeds row count"):
        shard_split.split_jsonl(src, 4, tmp_path / "o")


def test_shard_split_duplicate_keys_errors(tmp_path):
    # Duplicate stable keys are always rejected. Depending on whether the
    # dupes land in the same or different shards, the failure surfaces as
    # either an overlap (cross-shard) or a key-uniqueness (same-shard)
    # error -- both are correct disjointness rejections.
    src = tmp_path / "dup.jsonl"
    with src.open("w") as f:
        for i in [0, 1, 1, 2]:  # duplicate _attempt_idx=1 -> cross-shard
            f.write(json.dumps({"_attempt_idx": i, "messages": []}) + "\n")
    with pytest.raises(RuntimeError, match="overlap|KEY-UNIQUENESS"):
        shard_split.split_jsonl(src, 2, tmp_path / "o")

    # Same-shard duplicates (both copies in shard 0) trip key-uniqueness.
    src2 = tmp_path / "dup2.jsonl"
    with src2.open("w") as f:
        for i in [0, 0, 1, 2]:  # duplicate _attempt_idx=0 in shard 0
            f.write(json.dumps({"_attempt_idx": i, "messages": []}) + "\n")
    with pytest.raises(RuntimeError, match="overlap|KEY-UNIQUENESS"):
        shard_split.split_jsonl(src2, 2, tmp_path / "o2")


def test_shard_split_seed_idx_fallback(tmp_path):
    src = tmp_path / "seed.jsonl"
    _write_jsonl(src, 6, key="seed_idx")
    counts = shard_split.split_jsonl(src, 3, tmp_path / "o")
    assert counts == [2, 2, 2]


# ===========================================================================
# merge_sidecars: build a KNOWN single-run partition, then verify the merge
# of per-shard payloads reproduces the single-run result EXACTLY.
# ===========================================================================
N_LAYERS = 3
N_EXPERTS = 5


def _sidecar_dir_for(jsonl: Path) -> Path:
    return jsonl.parent / "sidecars" / jsonl.stem


def test_merge_reap_equals_single_run(tmp_path):
    """The probe-critical test: merged reap_scores == single full run.

    Construct a known per-(layer,expert) token partition split across 2
    shards. The single-run reap mean is score_sum / total_count. Each shard's
    payload stores its OWN mean (shard_score_sum / shard_count). The merge
    must reconstruct the global mean exactly.
    """
    torch.manual_seed(0)
    # Per-(l,e): split total tokens into two shard counts (some cells get 0
    # tokens in one shard to exercise the clamp path).
    count_a = torch.randint(0, 7, (N_LAYERS, N_EXPERTS), dtype=torch.int64)
    count_b = torch.randint(0, 7, (N_LAYERS, N_EXPERTS), dtype=torch.int64)
    # Force one cell to be all-zero in both shards (never routed).
    count_a[0, 0] = 0
    count_b[0, 0] = 0

    # Per-cell score sums (the underlying Σ g_j·||f_j|| the writer accumulates).
    score_sum_a = torch.rand(N_LAYERS, N_EXPERTS, dtype=torch.float64) * 13.0
    score_sum_b = torch.rand(N_LAYERS, N_EXPERTS, dtype=torch.float64) * 13.0
    # Zero score where count is zero (no tokens -> no contribution).
    score_sum_a = torch.where(count_a > 0, score_sum_a, torch.zeros_like(score_sum_a))
    score_sum_b = torch.where(count_b > 0, score_sum_b, torch.zeros_like(score_sum_b))

    # Single full run: global mean.
    total_count = count_a + count_b
    total_score = score_sum_a + score_sum_b
    single_run_mean = (
        total_score / total_count.clamp(min=1).to(torch.float64)
    ).to(torch.float32)

    # Each shard's payload stores its own mean (= score_sum / count).
    mean_a = (score_sum_a / count_a.clamp(min=1).to(torch.float64)).to(torch.float32)
    mean_b = (score_sum_b / count_b.clamp(min=1).to(torch.float64)).to(torch.float32)

    # Round-trip through the real save_*/load_* via per-shard sidecar dirs.
    shard_dirs = []
    for tag, mean, count in [("a", mean_a, count_a), ("b", mean_b, count_b)]:
        sj = tmp_path / f"shard_{tag}.jsonl"
        sj.write_text("")  # sidecar_path only needs the path, not contents.
        save_reap_scores(
            Stage2ReapPayload(
                schema_version=SCHEMA_VERSIONS["reap_scores"],
                n_experts=N_EXPERTS,
                n_layers=N_LAYERS,
                reap_scores=mean,
                token_counts=count,
            ),
            sj,
        )
        shard_dirs.append(_sidecar_dir_for(sj))

    out_jsonl = tmp_path / "merged_8000.jsonl"
    out_jsonl.write_text("")
    status = ms.merge_all(shard_dirs, out_jsonl)
    assert status["reap_scores"] == "merged"

    merged = load_reap_scores(out_jsonl)
    assert merged is not None
    assert tuple(merged.reap_scores.shape) == (N_LAYERS, N_EXPERTS)

    # EXACT equivalence (fp32 round-trip tolerance).
    assert torch.allclose(
        merged.reap_scores, single_run_mean, atol=1e-5, rtol=1e-5,
    ), (
        f"merged reap != single-run mean.\n"
        f"max abs diff = "
        f"{(merged.reap_scores - single_run_mean).abs().max().item()}"
    )
    # token_counts == total exactly.
    assert torch.equal(merged.token_counts, total_count)
    assert int(merged.token_counts.sum().item()) == int(total_count.sum().item())


def test_merge_per_expert_max(tmp_path):
    """Element-wise max across shards; counts summed."""
    torch.manual_seed(1)
    pem_a = torch.rand(N_LAYERS, N_EXPERTS, dtype=torch.float32) * 10
    pem_b = torch.rand(N_LAYERS, N_EXPERTS, dtype=torch.float32) * 10
    cnt_a = torch.randint(1, 5, (N_LAYERS, N_EXPERTS), dtype=torch.int64)
    cnt_b = torch.randint(1, 5, (N_LAYERS, N_EXPERTS), dtype=torch.int64)

    expected_max = torch.maximum(pem_a, pem_b)
    expected_cnt = cnt_a + cnt_b

    shard_dirs = []
    for tag, pem, cnt in [("a", pem_a, cnt_a), ("b", pem_b, cnt_b)]:
        sj = tmp_path / f"pem_{tag}.jsonl"
        sj.write_text("")
        save_per_expert_max(
            Stage1PerExpertMaxPayload(
                schema_version=SCHEMA_VERSIONS["per_expert_max"],
                n_experts=N_EXPERTS,
                n_layers=N_LAYERS,
                per_expert_max=pem,
                token_counts=cnt,
            ),
            sj,
        )
        shard_dirs.append(_sidecar_dir_for(sj))

    out_jsonl = tmp_path / "pem_merged.jsonl"
    out_jsonl.write_text("")
    ms.merge_all(shard_dirs, out_jsonl)
    merged = load_per_expert_max(out_jsonl)
    assert merged is not None
    assert torch.allclose(merged.per_expert_max, expected_max, atol=1e-6)
    assert torch.equal(merged.token_counts, expected_cnt)


def test_merge_routing_stats_weighted_mean(tmp_path):
    """freq summed; mean_weight is freq-weighted mean == single-run mean."""
    torch.manual_seed(2)
    freq_a = torch.randint(0, 8, (N_LAYERS, N_EXPERTS), dtype=torch.int64)
    freq_b = torch.randint(0, 8, (N_LAYERS, N_EXPERTS), dtype=torch.int64)
    freq_a[0, 0] = 0
    freq_b[0, 0] = 0

    # Underlying weight sums.
    wsum_a = torch.rand(N_LAYERS, N_EXPERTS, dtype=torch.float64) * 3
    wsum_b = torch.rand(N_LAYERS, N_EXPERTS, dtype=torch.float64) * 3
    wsum_a = torch.where(freq_a > 0, wsum_a, torch.zeros_like(wsum_a))
    wsum_b = torch.where(freq_b > 0, wsum_b, torch.zeros_like(wsum_b))

    total_freq = freq_a + freq_b
    single_mean = (
        (wsum_a + wsum_b) / total_freq.clamp(min=1).to(torch.float64)
    ).to(torch.float32)

    mean_a = (wsum_a / freq_a.clamp(min=1).to(torch.float64)).to(torch.float32)
    mean_b = (wsum_b / freq_b.clamp(min=1).to(torch.float64)).to(torch.float32)

    shard_dirs = []
    for tag, mw, fr in [("a", mean_a, freq_a), ("b", mean_b, freq_b)]:
        sj = tmp_path / f"rts_{tag}.jsonl"
        sj.write_text("")
        save_routing_stats(
            RoutingStatsPayload(
                schema_version=SCHEMA_VERSIONS["routing_stats"],
                n_experts=N_EXPERTS,
                n_layers=N_LAYERS,
                freq=fr,
                mean_weight=mw,
            ),
            sj,
        )
        shard_dirs.append(_sidecar_dir_for(sj))

    out_jsonl = tmp_path / "rts_merged.jsonl"
    out_jsonl.write_text("")
    ms.merge_all(shard_dirs, out_jsonl)
    merged = load_routing_stats(out_jsonl)
    assert merged is not None
    assert torch.equal(merged.freq, total_freq)
    assert torch.allclose(merged.mean_weight, single_mean, atol=1e-5, rtol=1e-5)


def test_merge_dim_mismatch_aborts(tmp_path):
    """Shards disagreeing on n_experts must abort."""
    sj_a = tmp_path / "a.jsonl"
    sj_a.write_text("")
    save_reap_scores(
        Stage2ReapPayload(
            schema_version=SCHEMA_VERSIONS["reap_scores"],
            n_experts=N_EXPERTS, n_layers=N_LAYERS,
            reap_scores=torch.zeros(N_LAYERS, N_EXPERTS),
            token_counts=torch.zeros(N_LAYERS, N_EXPERTS, dtype=torch.int64),
        ),
        sj_a,
    )
    sj_b = tmp_path / "b.jsonl"
    sj_b.write_text("")
    save_reap_scores(
        Stage2ReapPayload(
            schema_version=SCHEMA_VERSIONS["reap_scores"],
            n_experts=N_EXPERTS + 1, n_layers=N_LAYERS,
            reap_scores=torch.zeros(N_LAYERS, N_EXPERTS + 1),
            token_counts=torch.zeros(N_LAYERS, N_EXPERTS + 1, dtype=torch.int64),
        ),
        sj_b,
    )
    with pytest.raises(ValueError, match="disagree on n_experts"):
        ms.merge_all(
            [_sidecar_dir_for(sj_a), _sidecar_dir_for(sj_b)],
            tmp_path / "out.jsonl",
        )


def test_merge_absent_signal_skipped(tmp_path):
    """A signal absent from all shards is reported 'absent', not crashed."""
    sj = tmp_path / "only_reap.jsonl"
    sj.write_text("")
    save_reap_scores(
        Stage2ReapPayload(
            schema_version=SCHEMA_VERSIONS["reap_scores"],
            n_experts=N_EXPERTS, n_layers=N_LAYERS,
            reap_scores=torch.zeros(N_LAYERS, N_EXPERTS),
            token_counts=torch.ones(N_LAYERS, N_EXPERTS, dtype=torch.int64),
        ),
        sj,
    )
    out = tmp_path / "out.jsonl"
    out.write_text("")
    status = ms.merge_all([_sidecar_dir_for(sj)], out)
    assert status["reap_scores"] == "merged"
    assert status["per_expert_max"] == "absent"
    assert status["routing_stats"] == "absent"


# ===========================================================================
# N3: merge_block_outputs — per-layer concat across shards + layer UNION (H1).
# ===========================================================================
def test_merge_block_outputs_union_and_concat(tmp_path):
    """Per-layer block_hidden round-trip across 2 shards.

    Shard A has layers {0, 1}; shard B has layers {1, 2}. The merge must:
      * cover the UNION {0, 1, 2} (H1 — layer present only on one shard is
        NOT dropped),
      * concatenate hidden_states in shard order for layers seen in both,
      * sum n_prompts_in_subset per layer.
    """
    H = 4

    def _bh(layer, rows, n_prompts, fill):
        return BlockHiddenPayload(
            schema_version=SCHEMA_VERSIONS["block_hidden"],
            layer_idx=layer,
            n_prompts_in_subset=n_prompts,
            hidden_states=torch.full((rows, H), float(fill), dtype=torch.bfloat16),
        )

    sj_a = tmp_path / "bo_a.jsonl"; sj_a.write_text("")
    sj_b = tmp_path / "bo_b.jsonl"; sj_b.write_text("")

    # Shard A: layer 0 (2 rows, fill 1), layer 1 (3 rows, fill 2).
    save_block_hidden(_bh(0, 2, 5, 1), sj_a)
    save_block_hidden(_bh(1, 3, 5, 2), sj_a)
    # Shard B: layer 1 (1 row, fill 3), layer 2 (4 rows, fill 4).
    save_block_hidden(_bh(1, 1, 7, 3), sj_b)
    save_block_hidden(_bh(2, 4, 7, 4), sj_b)

    out = tmp_path / "bo_merged.jsonl"; out.write_text("")
    n_merged = ms.merge_block_outputs([sj_a, sj_b], out)
    assert n_merged == 3, f"expected union of 3 layers, got {n_merged}"

    # Layer 0: only shard A -> 2 rows, n_prompts 5.
    l0 = load_block_hidden(out, 0)
    assert l0 is not None and l0.hidden_states.shape == (2, H)
    assert l0.n_prompts_in_subset == 5

    # Layer 1: both shards, concat in shard order (A then B) -> 3+1=4 rows.
    l1 = load_block_hidden(out, 1)
    assert l1 is not None and l1.hidden_states.shape == (4, H)
    assert l1.n_prompts_in_subset == 12  # 5 + 7
    # First 3 rows == fill 2 (shard A), last row == fill 3 (shard B).
    hs = l1.hidden_states.to(torch.float32)
    assert torch.all(hs[:3] == 2.0)
    assert torch.all(hs[3:] == 3.0)

    # Layer 2: only shard B (NOT dropped — H1) -> 4 rows, n_prompts 7.
    l2 = load_block_hidden(out, 2)
    assert l2 is not None and l2.hidden_states.shape == (4, H)
    assert l2.n_prompts_in_subset == 7


# ===========================================================================
# H2: self-merge guard.
# ===========================================================================
def test_merge_self_merge_guard(tmp_path):
    """Output sidecar dir == an input shard dir must abort."""
    sj = tmp_path / "self.jsonl"; sj.write_text("")
    save_reap_scores(
        Stage2ReapPayload(
            schema_version=SCHEMA_VERSIONS["reap_scores"],
            n_experts=N_EXPERTS, n_layers=N_LAYERS,
            reap_scores=torch.zeros(N_LAYERS, N_EXPERTS),
            token_counts=torch.ones(N_LAYERS, N_EXPERTS, dtype=torch.int64),
        ),
        sj,
    )
    # out_jsonl == the same jsonl -> its sidecar dir == the input shard dir.
    with pytest.raises(ValueError, match="self-merge"):
        ms.merge_all([_sidecar_dir_for(sj)], sj)


# ===========================================================================
# concat_jsonls (generate-mode merge) disjointness.
# ===========================================================================
def test_concat_jsonls_disjoint_complete(tmp_path):
    a = tmp_path / "shard_a.jsonl"
    b = tmp_path / "shard_b.jsonl"
    _write_jsonl(a, 0)  # create empty then overwrite with explicit keys
    with a.open("w") as f:
        for i in [0, 1, 2]:
            f.write(json.dumps({"_attempt_idx": i, "messages": []}) + "\n")
    with b.open("w") as f:
        for i in [3, 4]:
            f.write(json.dumps({"_attempt_idx": i, "messages": []}) + "\n")

    out = tmp_path / "corpus.jsonl"
    n = ms.concat_jsonls([a, b], out)
    assert n == 5
    keys = [json.loads(l)["_attempt_idx"] for l in out.read_text().splitlines()]
    # shard order preserved (a then b), disjoint+complete.
    assert keys == [0, 1, 2, 3, 4]


def test_concat_jsonls_overlap_aborts(tmp_path):
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    with a.open("w") as f:
        for i in [0, 1, 2]:
            f.write(json.dumps({"_attempt_idx": i, "messages": []}) + "\n")
    with b.open("w") as f:
        for i in [2, 3]:  # key 2 overlaps shard a
            f.write(json.dumps({"_attempt_idx": i, "messages": []}) + "\n")
    out = tmp_path / "corpus.jsonl"
    with pytest.raises(RuntimeError, match="DISJOINTNESS"):
        ms.concat_jsonls([a, b], out)
    assert not out.exists(), "corpus must NOT be written on overlap"


# ===========================================================================
# generate-mode offset slices driven through the REAL buffer-dependent shuffle.
#
# This replaces the earlier tautological list-slice test. It drives a FAITHFUL
# fake of HF streaming ``shuffle(buffer_size=B)`` whose emit ORDER DEPENDS on B
# (a reservoir-style buffer shuffle), exactly the dependency that caused the
# HIGH bug: the driver's buffer = min(max(10000, 10*count), 200000) is
# count-DEPENDENT, so two processes with different --num-prompts get different
# orders. The test asserts:
#   * with a SHARED buffer (the --shuffle-buffer fix), the per-process
#     [prev_count, count) offset slices are disjoint + complete;
#   * with the OLD per-process count-derived buffer, the slices MISALIGN
#     (overlap and/or gap) — i.e. the test FAILS the disjoint+complete
#     property, proving it would catch a regression of the HIGH fix.
# ===========================================================================
import random as _random


def _fake_shuffled_stream(source, buffer_size, seed):
    """Faithful-ish model of HF ``IterableDataset.shuffle(buffer_size=B)``.

    Reservoir-buffer shuffle: fill a buffer of B items from the source, then
    repeatedly emit a RANDOM item from the buffer (seeded) and refill from the
    source. The emit ORDER is a deterministic function of (source, B, seed):
    same B -> same order; different B -> different order. This is the exact
    property the real ``ds.shuffle(buffer_size=...)`` has and which the HIGH
    bug hinges on.
    """
    rng = _random.Random(seed)
    src = iter(source)
    buf = []
    try:
        for _ in range(buffer_size):
            buf.append(next(src))
    except StopIteration:
        pass
    out = []
    while buf:
        j = rng.randrange(len(buf))
        out.append(buf[j])
        try:
            buf[j] = next(src)
        except StopIteration:
            buf.pop(j)
    return out


def _driver_offset_slice(source, seed, count, prev_count, *, shuffle_buffer):
    """Model ONE process: pull a buffer-B shuffle, take the first ``count``
    emitted, then yield positions [prev_count, count) — exactly the driver's
    per-subset offset semantics. ``shuffle_buffer`` None => count-derived
    (the OLD divergent behavior); set => shared (the fix)."""
    b = (shuffle_buffer if shuffle_buffer is not None
         else min(max(10_000, 10 * count), 200_000))
    order = _fake_shuffled_stream(source, b, seed)
    return order[prev_count:count]


def test_generate_offset_slices_disjoint_complete_shared_buffer():
    # Source MUST exceed the largest per-process buffer so reservoir size
    # actually affects emit order (a buffer >= source length degenerates to a
    # full in-memory shuffle, masking buffer differences). Largest old buffer
    # is 10*total = 45000, so source = 50000 keeps every buffer < source.
    source = list(range(50_000))
    seed = 1337
    n = 3
    total = 4500  # C_N
    ladder = [total * k // n for k in range(n + 1)]  # [0,1500,3000,4500]

    # SHARED buffer (the fix): one value for all processes.
    shared_buf = min(max(10_000, 10 * total), 200_000)
    slices = [
        _driver_offset_slice(
            source, seed, ladder[k + 1], ladder[k],
            shuffle_buffer=shared_buf,
        )
        for k in range(n)
    ]
    seen = set()
    for s in slices:
        sset = set(s)
        assert seen.isdisjoint(sset), "shared-buffer slices overlap"
        seen |= sset
    # Complete: union == the first C_N of the SHARED order, exactly.
    expected_order = _fake_shuffled_stream(source, shared_buf, seed)[:total]
    assert seen == set(expected_order)
    assert sum(len(s) for s in slices) == total


def test_generate_offset_slices_misalign_when_buffer_differs():
    # Same ladder, but each process uses the OLD count-derived buffer
    # (shuffle_buffer=None) -> per-process buffers differ -> orders diverge
    # -> slices MUST misalign (overlap OR gap). This is the regression guard:
    # if someone removes the --shuffle-buffer fix, this asserts the breakage.
    source = list(range(50_000))
    seed = 1337
    n = 3
    total = 4500
    ladder = [total * k // n for k in range(n + 1)]

    slices = [
        _driver_offset_slice(
            source, seed, ladder[k + 1], ladder[k],
            shuffle_buffer=None,  # OLD divergent behavior
        )
        for k in range(n)
    ]
    # Detect misalignment: either an overlap (a key in two slices) OR a gap
    # (union != the first C_N of ANY single consistent order, AND count<total
    # of distinct). With divergent orders the slices are NOT a clean partition.
    multiset_len = sum(len(s) for s in slices)
    union = set().union(*[set(s) for s in slices])
    overlap = multiset_len != len(union)
    # A clean partition of C_N distinct items would have len(union)==total with
    # no overlap; divergent buffers break at least one of these.
    clean_partition = (not overlap) and (len(union) == total)
    assert not clean_partition, (
        "count-derived (divergent) buffers unexpectedly produced a clean "
        "partition — the buffer-dependence the HIGH fix guards against did "
        "not manifest; the test can no longer detect a regression."
    )


def test_generate_ladder_uneven_split():
    # 10 prompts over 3 shards -> ladder [0,3,6,10]; last slice absorbs remainder.
    n, total = 3, 10
    ladder = [total * k // n for k in range(n + 1)]
    assert ladder == [0, 3, 6, 10]
    sizes = [ladder[k + 1] - ladder[k] for k in range(n)]
    assert sizes == [3, 3, 4]
    assert sum(sizes) == total


def test_concat_jsonls_gap_aborts():
    # A pure GAP: two internally-disjoint shards that together miss a prompt.
    # Disjointness (b) passes; only the expected_total guard catches it.
    a = tmp = None
    import tempfile
    d = Path(tempfile.mkdtemp())
    a = d / "a.jsonl"; b = d / "b.jsonl"
    with a.open("w") as f:
        for i in [0, 1]:
            f.write(json.dumps({"_attempt_idx": i, "messages": []}) + "\n")
    with b.open("w") as f:
        for i in [2, 3]:  # key 4 is the dropped/gapped prompt
            f.write(json.dumps({"_attempt_idx": i, "messages": []}) + "\n")
    out = d / "corpus.jsonl"
    # Without expected_total: passes (disjoint + complete-by-union).
    n = ms.concat_jsonls([a, b], out)
    assert n == 4
    # With expected_total=5 (the ladder expected 5): GAP caught.
    out.unlink()
    with pytest.raises(RuntimeError, match="COMPLETENESS FAILURE .gap."):
        ms.concat_jsonls([a, b], out, expected_total=5)
    assert not out.exists()
