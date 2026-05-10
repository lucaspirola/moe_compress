"""Tests for HF Hub upload + resume query (LLR-0030, LLR-0033).

We mock `huggingface_hub.HfApi` so the tests don't touch the network.

# VERIFIES: LLR-0030
# VERIFIES: LLR-0033
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from kdr.io.resume import (
    SAVE_COMPLETE_SENTINEL,
    download_partial_from_hub,
    find_latest_partial_on_hub,
    upload_final_to_hub,
    upload_partial_to_hub,
)


def _make_complete_partial(tmp_path: Path, name: str) -> Path:
    """Create a partial dir with the sentinel file."""
    p = tmp_path / name
    p.mkdir()
    (p / SAVE_COMPLETE_SENTINEL).touch()
    (p / "model.safetensors").write_bytes(b"\x00" * 16)
    return p


# ─────────────────────────────────────────────────────────────────────────────
# upload_partial_to_hub (LLR-0033)
# ─────────────────────────────────────────────────────────────────────────────


def test_upload_partial_refuses_without_sentinel(tmp_path: Path) -> None:
    """Uploading a partial that lacks _SAVE_COMPLETE would corrupt resume
    state — must raise."""
    bad = tmp_path / "kdr_bf16_partial_step10"
    bad.mkdir()
    (bad / "model.safetensors").write_bytes(b"x")
    # No sentinel.

    with pytest.raises(ValueError, match=r"lacks .*_SAVE_COMPLETE"):
        upload_partial_to_hub(bad, "fake/repo", create_repo=False)


def test_upload_partial_invokes_upload_folder(tmp_path: Path) -> None:
    """Calls HfApi.upload_folder with the partial dir's name as path-in-repo."""
    p = _make_complete_partial(tmp_path, "kdr_bf16_partial_step50")
    fake_api = MagicMock()
    with patch("huggingface_hub.HfApi", return_value=fake_api):
        url = upload_partial_to_hub(p, "fake/partials", create_repo=False)
    fake_api.upload_folder.assert_called_once()
    kwargs = fake_api.upload_folder.call_args.kwargs
    assert kwargs["folder_path"] == str(p)
    assert kwargs["path_in_repo"] == "kdr_bf16_partial_step50"
    assert kwargs["repo_id"] == "fake/partials"
    assert kwargs["repo_type"] == "model"
    assert "huggingface.co/fake/partials" in url


def test_upload_partial_creates_repo_when_requested(tmp_path: Path) -> None:
    p = _make_complete_partial(tmp_path, "kdr_bf16_partial_step1")
    fake_api = MagicMock()
    with patch("huggingface_hub.HfApi", return_value=fake_api):
        upload_partial_to_hub(p, "fake/repo", create_repo=True)
    fake_api.create_repo.assert_called_once_with(
        "fake/repo", exist_ok=True, private=True, repo_type="model"
    )


# ─────────────────────────────────────────────────────────────────────────────
# upload_final_to_hub (LLR-0030)
# ─────────────────────────────────────────────────────────────────────────────


def test_upload_final_invokes_upload_folder(tmp_path: Path) -> None:
    final = tmp_path / "kdr_bf16_recovered"
    final.mkdir()
    (final / "config.json").write_text("{}")
    (final / SAVE_COMPLETE_SENTINEL).touch()  # required by the sentinel guard
    fake_api = MagicMock()
    with patch("huggingface_hub.HfApi", return_value=fake_api):
        url = upload_final_to_hub(final, "fake/recovered", create_repo=False)
    fake_api.upload_folder.assert_called_once()
    kwargs = fake_api.upload_folder.call_args.kwargs
    # Final upload has NO `path_in_repo` (uploads to repo root).
    assert "path_in_repo" not in kwargs
    assert kwargs["folder_path"] == str(final)
    assert "huggingface.co/fake/recovered" in url


def test_upload_final_refuses_without_sentinel(tmp_path: Path) -> None:
    """LLR-0030 AC + LLR-0029 sentinel invariant: a final dir without
    `_SAVE_COMPLETE` represents a crash between rename and sentinel write —
    must raise rather than publish a truncated artifact."""
    bad = tmp_path / "kdr_bf16_recovered"
    bad.mkdir()
    (bad / "config.json").write_text("{}")
    # No sentinel.

    with pytest.raises(ValueError, match=r"lacks .*_SAVE_COMPLETE"):
        upload_final_to_hub(bad, "fake/recovered", create_repo=False)


# ─────────────────────────────────────────────────────────────────────────────
# find_latest_partial_on_hub (LLR-0033)
# ─────────────────────────────────────────────────────────────────────────────


def _tree_entries(*paths: str) -> list[MagicMock]:
    """Build mock list_repo_tree entries (`RepoFile`-shaped: just a `.path` attr)."""
    return [MagicMock(path=p) for p in paths]


def test_find_latest_on_hub_picks_highest_step_with_sentinel() -> None:
    """LLR-0033: among complete partials, pick the highest-step."""
    fake_api = MagicMock()
    fake_api.list_repo_tree.return_value = _tree_entries(
        "kdr_bf16_partial_step10/model.safetensors",
        f"kdr_bf16_partial_step10/{SAVE_COMPLETE_SENTINEL}",
        "kdr_bf16_partial_step50/model.safetensors",
        f"kdr_bf16_partial_step50/{SAVE_COMPLETE_SENTINEL}",
        "kdr_bf16_partial_step100/model.safetensors",
        # step100 lacks sentinel — must be skipped.
    )
    with patch("huggingface_hub.HfApi", return_value=fake_api):
        result = find_latest_partial_on_hub("fake/repo")
    assert result is not None
    name, step = result
    assert name == "kdr_bf16_partial_step50"
    assert step == 50
    # Verify the call used recursive=True (per LLR-0033 listing semantics).
    kwargs = fake_api.list_repo_tree.call_args.kwargs
    assert kwargs.get("recursive") is True


def test_find_latest_on_hub_returns_none_when_no_complete_partials() -> None:
    """LLR-0033 AC #2: a repo with no `_SAVE_COMPLETE`-d partials → None."""
    fake_api = MagicMock()
    fake_api.list_repo_tree.return_value = _tree_entries(
        "kdr_bf16_partial_step10/model.safetensors",
        # No sentinel.
        "README.md",
    )
    with patch("huggingface_hub.HfApi", return_value=fake_api):
        result = find_latest_partial_on_hub("fake/repo")
    assert result is None


def test_find_latest_on_hub_returns_none_for_missing_repo() -> None:
    """LLR-0033 AC #1: a repo that doesn't exist → None (not an error;
    trainer starts from step 0)."""
    from huggingface_hub.errors import RepositoryNotFoundError

    fake_api = MagicMock()
    # `RepositoryNotFoundError` is an `HfHubHTTPError` subclass — newer
    # versions require a `response` kwarg. Construct via `__new__` to
    # bypass __init__ for the unit test (we only need the exception type
    # to match the `except` branch).
    err = RepositoryNotFoundError.__new__(RepositoryNotFoundError)
    fake_api.list_repo_tree.side_effect = err
    with patch("huggingface_hub.HfApi", return_value=fake_api):
        result = find_latest_partial_on_hub("nonexistent/repo")
    assert result is None


def test_find_latest_on_hub_filters_non_partial_dirs() -> None:
    """Files whose top-level dir doesn't match `kdr_*_partial_step*` are skipped."""
    fake_api = MagicMock()
    fake_api.list_repo_tree.return_value = _tree_entries(
        "kdr_bf16_partial_step5/_SAVE_COMPLETE",
        "kdr_bf16_partial_step5/model.safetensors",
        "kdr_bf16_recovered/_SAVE_COMPLETE",  # final, not partial — skip
        "kdr_bf16_recovered/model.safetensors",
        "README.md",
    )
    with patch("huggingface_hub.HfApi", return_value=fake_api):
        result = find_latest_partial_on_hub("fake/repo")
    assert result is not None
    assert result[0] == "kdr_bf16_partial_step5"


def test_find_latest_on_hub_swallows_http_errors() -> None:
    """A transient 5xx mid-listing → return None (not crash)."""
    from huggingface_hub.errors import HfHubHTTPError

    fake_api = MagicMock()
    err = HfHubHTTPError.__new__(HfHubHTTPError)
    fake_api.list_repo_tree.side_effect = err
    with patch("huggingface_hub.HfApi", return_value=fake_api):
        result = find_latest_partial_on_hub("flaky/repo")
    assert result is None


# ─────────────────────────────────────────────────────────────────────────────
# download_partial_from_hub (LLR-0033)
# ─────────────────────────────────────────────────────────────────────────────


def test_download_partial_invokes_snapshot_download(tmp_path: Path) -> None:
    target = tmp_path / "artifacts"
    with patch("huggingface_hub.snapshot_download") as mock_dl:
        out = download_partial_from_hub(
            "fake/repo", "kdr_bf16_partial_step42", target
        )
    mock_dl.assert_called_once()
    kwargs = mock_dl.call_args.kwargs
    assert kwargs["repo_id"] == "fake/repo"
    assert kwargs["allow_patterns"] == ["kdr_bf16_partial_step42/*"]
    assert kwargs["local_dir"] == str(target)
    assert out == target / "kdr_bf16_partial_step42"
