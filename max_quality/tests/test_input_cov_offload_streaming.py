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
                ref[(li, e, "gate_proj")] = (cov[e].to(torch.float16), int(cnt[e]))
    return ref


def test_roundtrip_and_contract(tmp_path: Path):
    jsonl = _make_jsonl(tmp_path)
    staging = ico.staging_dir(jsonl)
    n_layers, n_experts, d = 3, 4, 5
    # layer 1 expert 2 gets zero tokens -> must be skipped everywhere.
    ref = _capture_layers(staging, n_layers, n_experts, d, empty=[(1, 2)])

    assert ico.scan_done_layers(staging) == {0, 1, 2}

    n = ico.assemble_covariance(staging, jsonl, n_experts, n_layers)
    assert n == len(ref)

    payload = load_covariance(jsonl)
    assert payload is not None
    assert payload.n_experts == n_experts and payload.n_layers == n_layers
    assert set(payload.sigma_in.keys()) == set(ref.keys())
    assert (1, 2, "gate_proj") not in payload.sigma_in  # empty expert skipped
    for k, (t_fp16, c) in ref.items():
        got = payload.sigma_in[k]
        assert got.dtype == torch.float16 and got.device.type == "cpu"
        assert tuple(got.shape) == (d, d)
        torch.testing.assert_close(got, t_fp16)         # exact fp16 match
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
    assert done == {0, 2}, done
    # Resume completes layer 1; assembly then sees all three.
    ico.write_layer_shard(staging, 1, torch.randn(n_experts, d, d),
                          torch.full((n_experts,), 5, dtype=torch.int64))
    assert ico.scan_done_layers(staging) == {0, 1, 2}
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


if __name__ == "__main__":
    import tempfile
    for fn in (test_roundtrip_and_contract, test_resume_scan_partial,
               test_atomic_shard_is_complete_or_absent,
               test_staging_dir_override_appends_subdir):
        with tempfile.TemporaryDirectory() as d:
            fn(Path(d))
    print("ALL PASS")
