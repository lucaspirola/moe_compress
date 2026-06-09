"""Unit tests for the streaming input_cov disk-offload (RAM-bomb fix).

CPU-only (no vLLM, no GPU): exercise the shard write -> scan -> assemble ->
load_covariance round-trip, the resume path (partial shards), and the on-disk
contract (fp16 sigma_in keyed (layer, expert, matrix), raw token_counts,
manifest present, empty experts skipped).

Run: PYTHONPATH=max_quality/src /usr/bin/python3 max_quality/tests/test_input_cov_offload_streaming.py
"""
import sys
from pathlib import Path

import torch

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from moe_compress.utils.cached_calibration_signals import load_covariance  # noqa: E402
from moe_compress.utils import input_cov_offload as ico  # noqa: E402


def _make_jsonl(tmp: Path) -> Path:
    j = tmp / "self_traces_deadbeef.jsonl"
    j.write_text('{"messages": []}\n')
    return j


def _capture_layers(staging: Path, n_layers: int, n_experts: int, d: int,
                    *, empty=( )):
    """Write synthetic per-layer Gram shards; return the expected reference."""
    ref: dict = {}
    torch.manual_seed(7)
    for li in range(n_layers):
        cov = torch.randn(n_experts, d, d, dtype=torch.float32)
        cnt = torch.tensor([10 * (e + 1) for e in range(n_experts)],
                           dtype=torch.int64)
        for (el, ee) in empty:
            if el == li:
                cnt[ee] = 0
        ico.write_layer_shard(staging, li, cov, cnt)
        for e in range(n_experts):
            if int(cnt[e]) > 0:
                ref[(li, e, "gate_proj")] = (cov[e].to(torch.bfloat16), int(cnt[e]))
    return ref


def test_roundtrip_and_contract(tmp_path: Path):
    jsonl = _make_jsonl(tmp_path)
    staging = ico.staging_dir(jsonl)
    n_layers, n_experts, d = 3, 4, 5
    # layer 1 expert 2 gets zero tokens -> must be skipped everywhere.
    ref = _capture_layers(staging, n_layers, n_experts, d, empty=[(1, 2)])

    assert ico.scan_done_layers(staging) == {
        (0, "gate_proj"), (1, "gate_proj"), (2, "gate_proj")}

    n = ico.assemble_covariance(staging, jsonl, n_experts, n_layers)
    assert n == len(ref)

    payload = load_covariance(jsonl)
    assert payload is not None
    assert payload.n_experts == n_experts and payload.n_layers == n_layers
    assert set(payload.sigma_in.keys()) == set(ref.keys())
    assert (1, 2, "gate_proj") not in payload.sigma_in  # empty expert skipped
    for k, (t_bf16, c) in ref.items():
        got = payload.sigma_in[k]
        assert got.dtype == torch.bfloat16 and got.device.type == "cpu"
        assert tuple(got.shape) == (d, d)
        torch.testing.assert_close(got, t_bf16)         # exact bf16 match
        assert payload.token_counts[k] == c             # raw, unnormalized
    print("  PASS  streaming round-trip + contract (fp16, keys, counts, empty-skip)")


def test_resume_scan_partial(tmp_path: Path):
    jsonl = _make_jsonl(tmp_path)
    staging = ico.staging_dir(jsonl)
    n_experts, d = 4, 5
    # Simulate a crash after layers 0 and 2 only.
    torch.manual_seed(1)
    for li in (0, 2):
        ico.write_layer_shard(staging, li,
                              torch.randn(n_experts, d, d),
                              torch.full((n_experts,), 5, dtype=torch.int64))
    done = ico.scan_done_layers(staging)
    assert done == {(0, "gate_proj"), (2, "gate_proj")}, done
    # Resume completes layer 1; assembly then sees all three.
    ico.write_layer_shard(staging, 1, torch.randn(n_experts, d, d),
                          torch.full((n_experts,), 5, dtype=torch.int64))
    assert ico.scan_done_layers(staging) == {
        (0, "gate_proj"), (1, "gate_proj"), (2, "gate_proj")}
    ico.assemble_covariance(staging, jsonl, n_experts, n_layers=3)
    payload = load_covariance(jsonl)
    assert payload is not None
    assert {li for (li, _e, _m) in payload.sigma_in} == {0, 1, 2}
    print("  PASS  resume scan (partial shards) + completion")


def test_atomic_shard_is_complete_or_absent(tmp_path: Path):
    """A present shard is fully loadable (atomic write); no torn shards."""
    jsonl = _make_jsonl(tmp_path)
    staging = ico.staging_dir(jsonl)
    ico.write_layer_shard(staging, 7, torch.randn(3, 4, 4),
                          torch.tensor([1, 0, 2]))
    p = ico.shard_path(staging, 7)
    assert p.exists()
    shard = torch.load(p, map_location="cpu", weights_only=False)
    assert shard["layer_idx"] == 7 and shard["schema"] == 1
    assert set(shard["sigma"].keys()) == {0, 2}        # expert 1 had 0 tokens
    assert shard["counts"] == {0: 1, 2: 2}
    print("  PASS  atomic shard complete + zero-token experts elided")


def test_staging_dir_override_appends_subdir(tmp_path: Path):
    """An explicit --input-cov-staging-dir must get a fixed _covariance_staging
    subdir appended, so a fresh-run rmtree never deletes the user's path."""
    jsonl = _make_jsonl(tmp_path)
    override = tmp_path / "shared_mount"
    s = ico.staging_dir(jsonl, str(override))
    assert s == override / "_covariance_staging"
    assert s != override  # never the bare user path (rmtree-safety)
    default = ico.staging_dir(jsonl)
    assert default.name == "_covariance_staging"
    print("  PASS  staging-dir override appends _covariance_staging (rmtree-safe)")


def test_ride_along_gate_and_down_shards_coexist(tmp_path: Path):
    """Ride-along: gate_proj + down_proj shards for the SAME layer must coexist
    (the matrix-aware shard_path fix) and assemble into ONE covariance.pt with
    both key sets. Distinct d_in (gate=5, down=3) confirms the shapes are kept
    separate per matrix."""
    jsonl = _make_jsonl(tmp_path)
    staging = ico.staging_dir(jsonl)
    n_experts, d_gate, d_down = 4, 5, 3
    torch.manual_seed(11)
    for li in range(2):
        gate = torch.randn(n_experts, d_gate, d_gate)
        down = torch.randn(n_experts, d_down, d_down)
        cnt = torch.full((n_experts,), 7, dtype=torch.int64)
        ico.write_layer_shard(staging, li, gate, cnt, matrix_name="gate_proj")
        ico.write_layer_shard(staging, li, down, cnt, matrix_name="down_proj")

    # Both shards present per layer -> distinct filenames, no clobber.
    assert ico.shard_path(staging, 0, "gate_proj").exists()
    assert ico.shard_path(staging, 0, "down_proj").exists()
    assert (ico.shard_path(staging, 0, "gate_proj")
            != ico.shard_path(staging, 0, "down_proj"))
    assert ico.scan_done_layers(staging) == {
        (0, "gate_proj"), (0, "down_proj"),
        (1, "gate_proj"), (1, "down_proj")}

    n = ico.assemble_covariance(staging, jsonl, n_experts, n_layers=2)
    assert n == 2 * 2 * n_experts          # 2 layers * 2 matrices * experts
    payload = load_covariance(jsonl)
    assert payload is not None
    gate_keys = {k for k in payload.sigma_in if k[2] == "gate_proj"}
    down_keys = {k for k in payload.sigma_in if k[2] == "down_proj"}
    assert len(gate_keys) == 2 * n_experts and len(down_keys) == 2 * n_experts
    # Per-matrix shapes preserved + bf16 storage.
    assert tuple(payload.sigma_in[(0, 0, "gate_proj")].shape) == (d_gate, d_gate)
    assert tuple(payload.sigma_in[(0, 0, "down_proj")].shape) == (d_down, d_down)
    assert payload.sigma_in[(0, 0, "down_proj")].dtype == torch.bfloat16
    print("  PASS  ride-along gate+down shards coexist + assemble both key sets")


def test_down_gather_grouping_matches_reference():
    """The down SYRK's NEW gather logic (triton_moe.py): because
    intermediate_cache2 is already per-(token, slot) token-major [R, I], rows
    are gathered DIRECTLY by argsort(_flat) (NOT _tok = order // top_k like the
    per-token gate path). Replicate the counting-sort + gather in pure torch
    (no .cu op) and assert each expert's contiguous block offsets[e]:offsets[e+1]
    equals exactly the rows routed to e (stable order), and the per-expert Gram
    X_e^T X_e matches the masked reference. This is the correctness linchpin."""
    torch.manual_seed(3)
    T, top_k, E, I = 6, 3, 4, 5
    topk_ids = torch.randint(0, E, (T, top_k), dtype=torch.int64)
    R = T * top_k
    # xmid row r corresponds to (token r//top_k, slot r%top_k) -> expert flat[r].
    xmid = torch.randn(R, I)
    flat = topk_ids.reshape(-1)                              # [R] row -> expert

    # --- replicate the in-graph prologue (counting-sort, no padding) ---
    counts = torch.zeros(E, dtype=torch.int64)
    counts.scatter_add_(0, flat, torch.ones(R, dtype=torch.int64))
    offsets = torch.zeros(E + 1, dtype=torch.int64)
    torch.cumsum(counts, 0, out=offsets[1:])
    order = torch.argsort(flat, stable=True)
    xs = torch.index_select(xmid, 0, order)                 # DIRECT gather

    # offsets[E] must equal R (every routed row contributes exactly once).
    assert int(offsets[-1]) == R
    for e in range(E):
        lo, hi = int(offsets[e]), int(offsets[e + 1])
        block = xs[lo:hi]                                   # gathered rows for e
        ref = xmid[flat == e]                               # rows routed to e
        # stable argsort preserves token order within an expert -> exact match.
        assert block.shape == ref.shape
        torch.testing.assert_close(block, ref)
        # the kernel accumulates cov[e] += X_e^T X_e over this block.
        torch.testing.assert_close(block.T @ block, ref.T @ ref)
    print("  PASS  down gather grouping matches reference (direct argsort, "
          "per-expert Gram)")


if __name__ == "__main__":
    import tempfile
    for fn in (test_roundtrip_and_contract, test_resume_scan_partial,
               test_atomic_shard_is_complete_or_absent,
               test_staging_dir_override_appends_subdir,
               test_ride_along_gate_and_down_shards_coexist):
        with tempfile.TemporaryDirectory() as d:
            fn(Path(d))
    test_down_gather_grouping_matches_reference()
    print("ALL PASS")
