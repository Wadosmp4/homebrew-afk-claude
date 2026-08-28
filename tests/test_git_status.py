"""Tests for companion/git_status.py (U10, R16) against a real git repo
fixture - no mocking of `git` itself, per the plan's own test scenarios.
"""
from __future__ import annotations

import subprocess

import pytest

from companion.git_status import get_diff, get_file_content, get_status, write_session_file


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


# --- get_file_content (Quick-Edit Changed Files plan, U1) -------------------


def test_get_file_content_for_a_real_changed_file_returns_its_exact_current_content(repo):
    (repo / "committed.txt").write_text("changed content\n")

    result = get_file_content(str(repo), "committed.txt")

    assert result.ok is True
    assert result.content == "changed content\n"
    assert result.reason is None


def test_get_file_content_matches_a_changed_files_relative_path_against_git_status(repo):
    """Covers KTD1: regression-guards the absolute-vs-relative bug found in
    doc review - a real changed file, addressed the normal way a mobile
    client would (a cwd-relative path), must be readable."""
    nested = repo / "src" / "app.py"
    nested.parent.mkdir()
    nested.write_text("print('new')\n")
    _git(repo, "add", "src/app.py")

    result = get_file_content(str(repo), "src/app.py")

    assert result.ok is True
    assert result.content == "print('new')\n"


def test_get_file_content_rejects_a_path_traversal_attempt_before_reading_anything(repo):
    outside_secret = repo.parent / "secret.txt"
    outside_secret.write_text("do not read me\n")

    result = get_file_content(str(repo), "../secret.txt")

    assert result.ok is False
    assert result.reason == "path_escape"
    assert result.content is None


def test_get_file_content_rejects_a_real_file_git_status_does_not_report_as_changed(repo):
    (repo / "untouched.txt").write_text("never modified\n")
    _git(repo, "add", "untouched.txt")
    _git(repo, "commit", "-q", "-m", "add untouched.txt")

    result = get_file_content(str(repo), "untouched.txt")

    assert result.ok is False
    assert result.reason == "not_changed"
    assert result.content is None


def test_get_file_content_flags_a_binary_changed_file_instead_of_returning_content(repo):
    (repo / "image.png").write_bytes(bytes([0x89, 0x50, 0x4E, 0x47, 0x00, 0x01, 0x02, 0x03]))

    result = get_file_content(str(repo), "image.png")

    assert result.ok is False
    assert result.reason == "binary"
    assert result.content is None


def test_get_file_content_rejects_a_file_git_status_reports_as_deleted(repo):
    (repo / "to_delete.txt").write_text("bye\n")
    _git(repo, "add", "to_delete.txt")
    _git(repo, "commit", "-q", "-m", "add to_delete.txt")
    (repo / "to_delete.txt").unlink()

    result = get_file_content(str(repo), "to_delete.txt")

    assert result.ok is False
    assert result.reason == "deleted"
    assert result.content is None


# --- write_session_file (Quick-Edit Changed Files plan, U2) -----------------


def test_write_session_file_succeeds_when_on_disk_content_matches_previous_content(repo):
    (repo / "committed.txt").write_text("changed\n")

    result = write_session_file(str(repo), "committed.txt", previous_content="changed\n", new_content="fixed\n")

    assert result.ok is True
    assert result.reason is None
    assert (repo / "committed.txt").read_text() == "fixed\n"


def test_write_session_file_rejects_a_conflict_and_leaves_the_file_untouched(repo):
    (repo / "committed.txt").write_text("changed on disk\n")

    result = write_session_file(
        str(repo), "committed.txt", previous_content="stale fetched content\n", new_content="fixed\n"
    )

    assert result.ok is False
    assert result.reason == "conflict"
    assert (repo / "committed.txt").read_text() == "changed on disk\n"


def test_write_session_file_rejects_a_path_traversal_attempt_without_touching_the_file(repo):
    outside_secret = repo.parent / "secret.txt"
    outside_secret.write_text("original\n")

    result = write_session_file(
        str(repo), "../secret.txt", previous_content="original\n", new_content="overwritten\n"
    )

    assert result.ok is False
    assert result.reason == "path_escape"
    assert outside_secret.read_text() == "original\n"


def test_write_session_file_rejects_a_file_not_reported_as_changed_without_touching_it(repo):
    (repo / "untouched.txt").write_text("never modified\n")
    _git(repo, "add", "untouched.txt")
    _git(repo, "commit", "-q", "-m", "add untouched.txt")

    result = write_session_file(
        str(repo), "untouched.txt", previous_content="never modified\n", new_content="sneaky edit\n"
    )

    assert result.ok is False
    assert result.reason == "not_changed"
    assert (repo / "untouched.txt").read_text() == "never modified\n"


def test_write_session_file_rejects_a_deleted_file_without_recreating_it(repo):
    (repo / "to_delete.txt").write_text("bye\n")
    _git(repo, "add", "to_delete.txt")
    _git(repo, "commit", "-q", "-m", "add to_delete.txt")
    (repo / "to_delete.txt").unlink()

    result = write_session_file(str(repo), "to_delete.txt", previous_content="bye\n", new_content="resurrected\n")

    assert result.ok is False
    assert result.reason == "deleted"
    assert not (repo / "to_delete.txt").exists()
