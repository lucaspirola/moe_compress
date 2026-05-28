"""F-C-1 regression: per-prompt teacher-logits .npz write must NOT produce
the `…/000.npz.tmp.npz` ghost file.

The pre-fix code did::

    tmp_fp = fp.with_suffix(fp.suffix + ".tmp")   # "000.npz.tmp"
    np.savez_compressed(tmp_fp, …)                # writes "000.npz.tmp.npz"
    os.replace(tmp_fp, fp)                        # raises FileNotFoundError

This test covers the helper the script now routes through
(``atomic_npz_save``), demonstrating that the ghost file no longer
appears and the write succeeds under the same call shape the script
uses.
"""
from __future__ import annotations

import numpy as np

from moe_compress.utils.atomic_io import atomic_npz_save


def test_calib_vllm_npz_no_ghost_file(tmp_path):
    """Mimic the script's call shape: write `000.npz` into a logits dir."""
    logits_dir = tmp_path / "self_traces_logits"
    logits_dir.mkdir()
    attempt_idx = 0
    fp = logits_dir / f"{int(attempt_idx):07d}.npz"
    atomic_npz_save(
        fp,
        token_ids=np.arange(16, dtype=np.int32),
        top_ids=np.zeros((16, 50), dtype=np.int32),
        top_logprobs=np.zeros((16, 50), dtype=np.float32),
        attempt_idx=np.int64(attempt_idx),
        top_k=np.int32(50),
    )
    # Final file exists and is loadable.
    assert fp.exists()
    with np.load(fp) as data:
        assert data["token_ids"].shape == (16,)
        assert data["top_ids"].shape == (16, 50)
        assert int(data["attempt_idx"]) == 0
        assert int(data["top_k"]) == 50

    # No ghost `000.npz.tmp.npz` (numpy's auto-suffix double-extension
    # bug that the old `np.savez_compressed(str_path, …)` flow produced).
    ghost = logits_dir / f"{int(attempt_idx):07d}.npz.tmp.npz"
    assert not ghost.exists(), (
        "F-C-1 regression: numpy's .npz auto-suffix bug re-introduced — "
        "the script wrote to '<file>.npz.tmp.npz' instead of '<file>.npz'."
    )
    # No leftover .tmp.
    assert not list(logits_dir.glob("*.tmp"))
    assert not list(logits_dir.glob("*.tmp.npz"))


def test_calib_vllm_npz_many_attempts(tmp_path):
    """Multiple writes (e.g. resume re-running prompts) don't leak."""
    logits_dir = tmp_path / "_logits"
    for i in range(5):
        fp = logits_dir / f"{i:07d}.npz"
        atomic_npz_save(
            fp,
            token_ids=np.array([i], dtype=np.int32),
            top_ids=np.array([[i, i + 1]], dtype=np.int32),
            top_logprobs=np.array([[0.0, -1.0]], dtype=np.float32),
            attempt_idx=np.int64(i),
            top_k=np.int32(2),
        )
    # All 5 land at their canonical names; no ghost files.
    final_files = sorted(p.name for p in logits_dir.glob("*.npz"))
    assert final_files == [f"{i:07d}.npz" for i in range(5)]
    assert not list(logits_dir.glob("*.tmp"))
    assert not list(logits_dir.glob("*.tmp.npz"))


def test_calib_vllm_npz_overwrite(tmp_path):
    """Resume scenario: re-running the same attempt overwrites cleanly."""
    fp = tmp_path / "000.npz"
    atomic_npz_save(fp, x=np.array([1, 2, 3], dtype=np.int32))
    atomic_npz_save(fp, x=np.array([7, 8, 9], dtype=np.int32))
    with np.load(fp) as d:
        assert np.array_equal(d["x"], np.array([7, 8, 9], dtype=np.int32))
