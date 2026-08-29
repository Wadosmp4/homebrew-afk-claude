"""Git plumbing for R16 (read-only status/diffs), scoped to one session's
working directory. Shells out to `git`'s read-only subcommands only -
`status`, `diff`, `log` - never a mutating command (KD7: view-only, no
commit/checkout/pull/push from the phone).

Pure functions operating on a `cwd` path, not the event model - the daemon
(companion/daemon.py) is what turns a result into a `git_status`/`git_diff`
event on the requesting session's stream (see its `_handle_action`).
"""
from __future__ import annotations

import os
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
        # code-review finding (P1, follow-on from the get_file_content/
        # write_session_file UnicodeDecodeError fix below): `text=True`
        # decodes stdout/stderr with the strict error handler by default -
        # a tracked file with non-UTF-8 bytes (get_diff's own diff output
        # embeds the file's actual content) would otherwise raise
        # UnicodeDecodeError straight out of subprocess.run, before
        # get_file_content's own binary check ever gets a chance to run,
        # let alone reach the open() call its except clause now covers.
        # `errors="replace"` keeps this call in the same "never raises for
        # file content, only for real OSError-shaped failures" contract
        # every other caller here already relies on - a decode-mangled
        # diff is still readable enough for the "Binary files"/binary-byte
        # heuristics that consume it.
        errors="replace",
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


@dataclass
class PathValidation:
    ok: bool
    resolved_path: Optional[str] = None
    # One of "path_escape" / "deleted" / "not_changed" when ok=False.
    reason: Optional[str] = None


def _validate_session_path(cwd: str, path: str) -> PathValidation:
    """Quick-Edit Changed Files plan, KTD1: shared by get_file_content and
    write_session_file - a path must both stay inside the session's own
    working directory (containment) and currently be one of the files
    `git status` reports as changed (membership) before either read or
    write touches it.

    Containment mirrors auto_approve.py's _edit_is_auto_approvable exactly:
    a relative path is joined with `cwd` *before* being resolved; an
    absolute path is resolved as-is. This is the one existing path-safety
    precedent anywhere in companion/ - git_diff's own `path` parameter has
    none, which is deliberately not the pattern followed here.

    Membership compares `os.path.relpath(resolved_path, resolved_cwd)`,
    forward-slash-normalized, for an *exact* match against get_status(cwd)'s
    modified/added lists - never a substring/endswith match, and never the
    absolute resolved path itself, since git's own porcelain output is
    always cwd-relative. A doc-review pass on this plan found the originally
    drafted check compared the absolute path directly, which would have
    rejected every legitimate changed file it was ever handed."""
    try:
        resolved_cwd = os.path.realpath(cwd)
        resolved_path = os.path.realpath(path if os.path.isabs(path) else os.path.join(cwd, path))
    except OSError:
        return PathValidation(ok=False, reason="path_escape")
    if not (resolved_path == resolved_cwd or resolved_path.startswith(resolved_cwd + os.sep)):
        return PathValidation(ok=False, reason="path_escape")

    status = get_status(cwd)
    relative = os.path.relpath(resolved_path, resolved_cwd).replace(os.sep, "/")
    if relative in status.deleted:
        return PathValidation(ok=False, reason="deleted")
    if relative not in status.modified and relative not in status.added:
        return PathValidation(ok=False, reason="not_changed")
    return PathValidation(ok=True, resolved_path=resolved_path)


@dataclass
class FileContentResult:
    ok: bool
    content: Optional[str] = None
    # One of "path_escape" / "deleted" / "not_changed" / "binary" /
    # "read_failed" when ok=False.
    reason: Optional[str] = None

    def to_dict(self) -> dict:
        return {"ok": self.ok, "content": self.content, "reason": self.reason}


def get_file_content(cwd: str, path: str) -> FileContentResult:
    """Quick-Edit Changed Files plan (R1, R4, R5, R8): the file's current
    on-disk content, once path containment and changed-file membership
    both hold. Binary detection (KTD5) reuses get_diff's own heuristic
    rather than a second one - a binary changed file is flagged, not read,
    matching the deleted-file rejection _validate_session_path already
    gives for a deleted one."""
    validation = _validate_session_path(cwd, path)
    if not validation.ok:
        return FileContentResult(ok=False, reason=validation.reason)
    if get_diff(cwd, path).is_binary:
        return FileContentResult(ok=False, reason="binary")
    try:
        with open(validation.resolved_path, "r", encoding="utf-8") as f:
            content = f.read()
    except (OSError, UnicodeDecodeError):
        # code-review finding (P1): a text file with non-UTF-8 bytes but no
        # NUL byte (so get_diff's own binary heuristic doesn't catch it)
        # raises UnicodeDecodeError here - a ValueError subclass, not an
        # OSError, so the original `except OSError` alone let it propagate
        # uncaught, get swallowed by daemon.py's generic exception handler,
        # and never reach the phone as the rejection event the contract
        # promises. Reported the same way any other unreadable file is.
        return FileContentResult(ok=False, reason="read_failed")
    return FileContentResult(ok=True, content=content)


@dataclass
class WriteFileResult:
    ok: bool
    # One of "path_escape" / "deleted" / "not_changed" / "conflict" /
    # "read_failed" / "write_failed" when ok=False.
    reason: Optional[str] = None

    def to_dict(self) -> dict:
        return {"ok": self.ok, "reason": self.reason}


def write_session_file(cwd: str, path: str, previous_content: str, new_content: str) -> WriteFileResult:
    """Quick-Edit Changed Files plan (R3, R4, R5, R6; KTD2): overwrites the
    file with `new_content`, full-content (no diff/patch), once path
    containment and changed-file membership both hold and the file's
    current on-disk content still matches `previous_content` byte-for-byte
    - the exact content the client fetched when editing began, not a hash.
    The read-compare-write below is back-to-back with no other I/O between
    the compare and the write, minimizing (not eliminating - see the
    plan's own Scope Boundaries) the race against a concurrent writer."""
    validation = _validate_session_path(cwd, path)
    if not validation.ok:
        return WriteFileResult(ok=False, reason=validation.reason)
    try:
        with open(validation.resolved_path, "r", encoding="utf-8") as f:
            current_content = f.read()
    except (OSError, UnicodeDecodeError):
        # code-review finding (P1): same UnicodeDecodeError gap as
        # get_file_content above - a non-UTF-8-but-no-NUL-byte file must
        # produce the read_failed rejection event, not an uncaught crash.
        return WriteFileResult(ok=False, reason="read_failed")
    if current_content != previous_content:
        return WriteFileResult(ok=False, reason="conflict")
    try:
        with open(validation.resolved_path, "w", encoding="utf-8") as f:
            f.write(new_content)
    except (OSError, UnicodeEncodeError):
        return WriteFileResult(ok=False, reason="write_failed")
    return WriteFileResult(ok=True)
