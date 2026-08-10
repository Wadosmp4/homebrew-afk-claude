"""Git plumbing for R16 (read-only status/diffs), scoped to one session's
working directory. Shells out to `git`'s read-only subcommands only -
`status`, `diff`, `log` - never a mutating command (KD7: view-only, no
commit/checkout/pull/push from the phone).

Pure functions operating on a `cwd` path, not the event model - the daemon
(companion/daemon.py) is what turns a result into a `git_status`/`git_diff`
event on the requesting session's stream (see its `_handle_action`).
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from typing import Optional

# Git's porcelain v1 status codes: X = index (staged), Y = worktree
# (unstaged). '?' means untracked. See `git help status` --porcelain.
_ADDED_CODES = {"A", "?"}
_DELETED_CODES = {"D"}


def _run_git(cwd: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=10,
    )


def _is_git_repo(cwd: str) -> bool:
    result = _run_git(cwd, "rev-parse", "--is-inside-work-tree")
    return result.returncode == 0 and result.stdout.strip() == "true"


@dataclass
class GitStatus:
    is_git_repo: bool
    branch: Optional[str] = None
    modified: list[str] = field(default_factory=list)
    added: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    last_commit: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "is_git_repo": self.is_git_repo,
            "branch": self.branch,
            "modified": self.modified,
            "added": self.added,
            "deleted": self.deleted,
            "last_commit": self.last_commit,
        }


def get_status(cwd: str) -> GitStatus:
    """R16: branch, changed/added/deleted files, last commit message.

    Returns `is_git_repo=False` (with everything else empty) rather than
    raising when `cwd` isn't inside a git repository - a session started
    outside version control is a normal case, not an error."""
    if not _is_git_repo(cwd):
        return GitStatus(is_git_repo=False)

    branch_result = _run_git(cwd, "rev-parse", "--abbrev-ref", "HEAD")
    branch = branch_result.stdout.strip() if branch_result.returncode == 0 else None

    commit_result = _run_git(cwd, "log", "-1", "--pretty=%s")
    last_commit = commit_result.stdout.strip() if commit_result.returncode == 0 and commit_result.stdout.strip() else None

    porcelain = _run_git(cwd, "status", "--porcelain")
    modified: list[str] = []
    added: list[str] = []
    deleted: list[str] = []
    for line in porcelain.stdout.splitlines():
        if not line:
            continue
        # Porcelain v1: two status chars, a space, then the path (possibly
        # `old -> new` for a rename, which we treat as a modification of
        # `new` - a full rename-aware UI is future work, not R16's scope).
        code = line[:2]
        path = line[3:].split(" -> ")[-1]
        if "?" in code or "A" in code:
            added.append(path)
        elif "D" in code:
            deleted.append(path)
        else:
            modified.append(path)

    return GitStatus(is_git_repo=True, branch=branch, modified=modified, added=added, deleted=deleted, last_commit=last_commit)


@dataclass
class GitDiff:
    is_git_repo: bool
    is_binary: bool = False
    diff: Optional[str] = None

    def to_dict(self) -> dict:
        return {"is_git_repo": self.is_git_repo, "is_binary": self.is_binary, "diff": self.diff}


def get_diff(cwd: str, path: str) -> GitDiff:
    """R16: the unified diff for one changed file, including an
    untracked file (diffed against `/dev/null` via `--no-index`).

    A binary file is reported as changed (is_binary=True) without
    attempting to render a text diff - git's own `Binary files differ`
    porcelain line is the signal, not a heuristic on the path/content."""
    if not _is_git_repo(cwd):
        return GitDiff(is_git_repo=False)

    tracked = _run_git(cwd, "diff", "HEAD", "--", path)
    output = tracked.stdout
    if not output.strip():
        # Not modified relative to HEAD - could be untracked; diff against
        # nothing so a brand-new file still gets a real unified diff.
        untracked = _run_git(cwd, "diff", "--no-index", "--", "/dev/null", path)
        output = untracked.stdout

    if "Binary files" in output or "\x00" in output:
        return GitDiff(is_git_repo=True, is_binary=True)

    return GitDiff(is_git_repo=True, is_binary=False, diff=output or None)
