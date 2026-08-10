"""Tests for companion/git_status.py (U10, R16) against a real git repo
fixture - no mocking of `git` itself, per the plan's own test scenarios.
"""
from __future__ import annotations

import subprocess

import pytest

from companion.git_status import get_diff, get_status


def _git(cwd, *args) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    _git(repo_dir, "init", "-q")
    _git(repo_dir, "config", "user.email", "test@example.com")
    _git(repo_dir, "config", "user.name", "Test")
    (repo_dir / "committed.txt").write_text("original\n")
    _git(repo_dir, "add", "committed.txt")
    _git(repo_dir, "commit", "-q", "-m", "Initial commit")
    return repo_dir


def test_status_on_non_git_directory_reports_not_a_repo(tmp_path):
    plain_dir = tmp_path / "not-a-repo"
    plain_dir.mkdir()

    status = get_status(str(plain_dir))

    assert status.is_git_repo is False
    assert status.modified == []
    assert status.branch is None


def test_status_reports_modified_added_and_deleted_files(repo):
    (repo / "committed.txt").write_text("changed\n")
    (repo / "new_file.txt").write_text("brand new\n")
    (repo / "to_delete.txt").write_text("bye\n")
    _git(repo, "add", "to_delete.txt")
    _git(repo, "commit", "-q", "-m", "add to_delete.txt")
    (repo / "to_delete.txt").unlink()

    status = get_status(str(repo))

    assert status.is_git_repo is True
    assert "committed.txt" in status.modified
    assert "new_file.txt" in status.added
    assert "to_delete.txt" in status.deleted


def test_status_reports_branch_and_last_commit_message(repo):
    status = get_status(str(repo))

    assert status.branch in ("main", "master")  # depends on git's init.defaultBranch
    assert status.last_commit == "Initial commit"


def test_status_matches_git_status_porcelain_output(repo):
    (repo / "committed.txt").write_text("changed\n")
    (repo / "untracked.txt").write_text("new\n")

    status = get_status(str(repo))
    porcelain = subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout

    assert " M committed.txt" in porcelain or "M  committed.txt" in porcelain
    assert "?? untracked.txt" in porcelain
    assert "committed.txt" in status.modified
    assert "untracked.txt" in status.added


def test_diff_on_non_git_directory_reports_not_a_repo(tmp_path):
    plain_dir = tmp_path / "not-a-repo"
    plain_dir.mkdir()

    diff = get_diff(str(plain_dir), "anything.txt")

    assert diff.is_git_repo is False
    assert diff.diff is None


def test_diff_for_a_modified_tracked_file_returns_a_unified_diff(repo):
    (repo / "committed.txt").write_text("changed content\n")

    diff = get_diff(str(repo), "committed.txt")

    assert diff.is_git_repo is True
    assert diff.is_binary is False
    assert "-original" in diff.diff
    assert "+changed content" in diff.diff


def test_diff_for_a_new_untracked_file_returns_a_diff_against_empty(repo):
    (repo / "new_file.txt").write_text("brand new content\n")

    diff = get_diff(str(repo), "new_file.txt")

    assert diff.is_git_repo is True
    assert diff.is_binary is False
    assert "brand new content" in diff.diff


def test_diff_for_a_binary_file_reports_binary_without_a_text_diff(repo):
    (repo / "image.png").write_bytes(bytes([0x89, 0x50, 0x4E, 0x47, 0x00, 0x01, 0x02, 0x03]))

    diff = get_diff(str(repo), "image.png")

    assert diff.is_git_repo is True
    assert diff.is_binary is True
    assert diff.diff is None
