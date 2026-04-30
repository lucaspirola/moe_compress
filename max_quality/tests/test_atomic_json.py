"""Test atomic JSON artifact writing."""
import json
import os
from pathlib import Path
import pytest
from moe_compress.utils.model_io import save_json_artifact, load_json_artifact


def test_save_json_artifact_atomic(tmp_path):
    """Verify no .tmp file is left on success."""
    p = tmp_path / "sub" / "out.json"
    save_json_artifact({"key": "value"}, p)
    assert p.exists()
    # No stray .tmp files
    assert not list(tmp_path.rglob("*.tmp"))
    assert load_json_artifact(p) == {"key": "value"}


def test_save_json_artifact_no_truncation(tmp_path):
    """Verify write is atomic: original survives if a new write fails mid-way."""
    p = tmp_path / "data.json"
    save_json_artifact({"version": 1}, p)
    import moe_compress.utils.model_io as mio
    calls = []
    real_replace = os.replace
    def spy_replace(src, dst):
        calls.append((src, dst))
        real_replace(src, dst)
    import unittest.mock as mock
    with mock.patch("moe_compress.utils.model_io.os.replace", spy_replace):
        save_json_artifact({"version": 2}, p)
    assert len(calls) == 1
    src, dst = calls[0]
    src_str = str(src)
    dst_str = str(dst)
    assert src_str.endswith(".tmp")
    assert not src_str.endswith(str(p))
    assert dst_str == str(p)
