"""Policy-based auto-approval for tool calls - opt-in per session, gated
on the phone at start_session time (R: "autoapprove ... if they are not
critical", "limit it very strictly", extended to cover "running tests or
similar" Bash and low-blast-radius edits).

Deliberately rule-based, not LLM-judged - see sdk_adapter.py's can_use_tool
for the reasoning: a hard-coded rule is auditable and has zero latency; an
LLM judge per tool call means spawning a subprocess on the hot path and
trusting an opaque classifier for a security-relevant decision. That's a
sharper tradeoff for Edit/Write specifically, where "is this a good code
change" is a real content-judgment question no static rule can answer - this
module answers a narrower, honestly-scoped question instead ("is this an
ordinary edit inside the project, not a sensitive file"), not "is the diff
itself good." An LLM judge for that remains a deliberate, not-yet-built v2.

Bash stays allowlist-based, never denylist-based: a command must match a
known-safe pattern to skip the prompt, not merely fail to match a forbidden
one - flipping that (allow by default, block known-bad) would be a real
increase in risk, since an arbitrary unmatched command could still do
anything. The denylist wins over any allowlist match unconditionally.
"""
from __future__ import annotations

import os
import re
from typing import Any, Optional

# Read-only tools: no state change is possible, so these are always safe to
# auto-approve regardless of arguments.
_ALWAYS_SAFE_TOOLS = frozenset({"Read", "Grep", "Glob", "WebSearch"})


def is_structured_question(tool_input: Optional[dict]) -> bool:
    """Recognizes a structured-choice tool call (AskUserQuestion) by input
    shape rather than tool name, matching the mobile client's own
    `parseStructuredQuestion` (EventFeed.tsx) - both sides need to agree on
    what counts as one. Shared by every caller of this policy (SDKAdapter's
    can_use_tool, ObserveAdapter's PermissionRequest hook handling): a
    structured question can never be auto-approved by the policy or the LLM
    judge, no matter which adapter owns the session - "is this tool call
    safe to run" is the wrong question for it, since the real answer isn't
    "run it" or "don't", it's the phone's chosen option text, which only
    the human-prompt path knows how to carry back (see each adapter's
    handling of a structured-question decision)."""
    if not isinstance(tool_input, dict):
        return False
    questions = tool_input.get("questions")
    return isinstance(questions, list) and len(questions) > 0

# Bash command patterns that must NEVER auto-approve, grouped by why each
# category is here - every entry is either the user's explicit concern
# (git actions that change remote/shared state) or shares its shape
# closely enough to warrant the same treatment. Checked before the
# allowlist below and wins unconditionally.
_DENYLIST_PATTERNS = [
    # Git: rewrites or changes remote/shared history, or discards local
    # work - the user's explicit "pushing/merging/changing remote branch
    # state" concern.
    re.compile(r"\bgit\s+push\b"),
    re.compile(r"\bgit\s+merge\b"),
    re.compile(r"\bgit\s+rebase\b"),
    re.compile(r"\bgit\s+reset\s+--hard\b"),
    re.compile(r"\bgit\s+branch\s+-[dD]\b"),
    re.compile(r"\bgit\s+checkout\b.*--force\b"),
    re.compile(r"\bgh\s+pr\s+merge\b"),  # GitHub CLI's equivalent of git merge
    re.compile(r"\bgh\s+repo\s+delete\b"),
    # Publishing: ships this project's code to a public registry - same
    # "remote, hard-to-undo state change" shape as git push, just for
    # package managers instead of git remotes.
    re.compile(r"\bnpm\s+publish\b"),
    re.compile(r"\byarn\s+publish\b"),
    re.compile(r"\btwine\s+upload\b"),
    re.compile(r"\bcargo\s+publish\b"),
    re.compile(r"\bgem\s+push\b"),
    # Infra/deploy: changes a real, running remote system, not just this
    # repo - the highest-consequence category short of the git/publish
    # ones above.
    re.compile(r"\bterraform\s+(apply|destroy)\b"),
    re.compile(r"\bkubectl\s+(apply|delete)\b"),
    re.compile(r"\bvercel\b.*--prod\b"),
    re.compile(r"\bnetlify\s+deploy\b.*--prod\b"),
    re.compile(r"\bfirebase\s+deploy\b"),
    # Destructive filesystem operations and privilege escalation.
    re.compile(r"\brm\s+-\w*[rf]\w*[rf]?\w*\b"),  # rm -rf / -fr / -Rf / etc.
    re.compile(r"\bsudo\b"),
    # Secrets/credentials - reading or writing these can leak or corrupt
    # auth material the user can't easily rotate from their phone. Not
    # `.env.example`/`.sample`/`.template`/`.dist` - those are ordinary,
    # secret-free committed files, not credentials, and denylisting them
    # was a real false positive worth fixing at the pattern level rather
    # than by letting anything (LLM or otherwise) override the denylist.
    re.compile(r"\.env\b(?!\.(example|sample|template|dist))"),
    re.compile(r"\bid_rsa\b|\.pem\b|\.ppk\b"),
    re.compile(r"\.aws/credentials\b|\.ssh/config\b"),
    # Arbitrary remote code execution.
    re.compile(r"\b(curl|wget)\b.*\|\s*(sh|bash|zsh)\b"),
]

# Bash commands that ARE auto-approvable, each genuinely non-destructive by
# strong convention rather than merely "looks safe": read-only git
# inspection, running tests, static analysis/type-checking (no mutating
# flag), and a local build that doesn't publish or deploy (see the
# denylist above for those).
_ALLOWLIST_PATTERNS = [
    re.compile(r"^\s*git\s+(status|diff|log|show|blame|branch\b|fetch)\b"),
    re.compile(r"\b(pytest|py\.test)\b"),
    re.compile(r"\bnpm\s+(run\s+)?test\b"),
    re.compile(r"\byarn\s+(run\s+)?test\b"),
    re.compile(r"\bgo\s+test\b"),
    re.compile(r"\bcargo\s+test\b"),
    re.compile(r"\bmake\s+test\b"),
    re.compile(r"\bbundle\s+exec\s+rspec\b"),
    re.compile(r"\b(eslint|flake8|mypy|tsc|ruff\s+check|black\s+--check)\b"),
    re.compile(r"\bnpm\s+run\s+build\b"),
    re.compile(r"\bgo\s+build\b"),
    re.compile(r"\bcargo\s+build\b"),
]

# A flag that turns an otherwise-allowlisted "check" command into a
# mutation (e.g. `eslint --fix`, `ruff check --fix`, `black` without
# `--check`'s read-only guarantee) - disqualifies the allowlist match even
# if the base command matched, since the command no longer has the
# property the allowlist entry was trusting.
_MUTATION_FLAGS = re.compile(r"--fix\b|--write\b|(?<!\S)-w\b")

# File path patterns Edit/Write/MultiEdit must never auto-approve, even
# when the toggle is on and the file is inside the project - same spirit
# as the Bash denylist's secrets/credentials entries, plus dependency
# lockfiles (a supply-chain-adjacent trust boundary) and CI/CD config.
_SENSITIVE_EDIT_PATTERNS = [
    re.compile(r"(^|/)\.env(\.|$)"),
    re.compile(r"(^|/)(package-lock\.json|yarn\.lock|pnpm-lock\.yaml|Cargo\.lock|Gemfile\.lock|poetry\.lock)$"),
    re.compile(r"(^|/)\.github/workflows/"),
    re.compile(r"(^|/)(id_rsa|[^/]+\.pem|[^/]+\.ppk)$"),
    re.compile(r"(^|/)\.ssh/"),
    re.compile(r"(^|/)\.aws/"),
]


def is_denylisted(tool_name: str, tool_input: Optional[dict[str, Any]]) -> bool:
    """True if this call must always prompt, no matter what - wins over
    is_auto_approvable unconditionally."""
    if tool_name != "Bash" or not isinstance(tool_input, dict):
        return False
    command = tool_input.get("command")
    if not isinstance(command, str):
        return False
    return any(pattern.search(command) for pattern in _DENYLIST_PATTERNS)


def _bash_is_auto_approvable(tool_input: Optional[dict[str, Any]]) -> bool:
    if not isinstance(tool_input, dict):
        return False
    command = tool_input.get("command")
    if not isinstance(command, str):
        return False
    if _MUTATION_FLAGS.search(command):
        return False
    return any(pattern.search(command) for pattern in _ALLOWLIST_PATTERNS)


def _edit_is_auto_approvable(tool_input: Optional[dict[str, Any]], cwd: Optional[str]) -> bool:
    if not isinstance(tool_input, dict):
        return False
    path = tool_input.get("file_path")
    if not isinstance(path, str) or not path:
        return False
    if any(pattern.search(path) for pattern in _SENSITIVE_EDIT_PATTERNS):
        return False
    if not cwd:
        # No known project scope to check "stays inside the project"
        # against - fail closed rather than approve blind.
        return False
    try:
        resolved_cwd = os.path.realpath(cwd)
        resolved_path = os.path.realpath(path if os.path.isabs(path) else os.path.join(cwd, path))
    except OSError:
        return False
    return resolved_path == resolved_cwd or resolved_path.startswith(resolved_cwd + os.sep)


def is_auto_approvable(
    tool_name: str, tool_input: Optional[dict[str, Any]], cwd: Optional[str] = None
) -> bool:
    """True for the set of calls this policy accepts as low-risk enough to
    skip the prompt: always for the read-only tools, for a Bash command
    matching the allowlist (and not the denylist), or for an Edit/Write/
    MultiEdit whose target file stays inside the project and isn't a
    sensitive path. Everything else - Bash outside the allowlist, WebFetch,
    Task - still prompts."""
    if is_denylisted(tool_name, tool_input):
        return False
    if tool_name in _ALWAYS_SAFE_TOOLS:
        return True
    if tool_name == "Bash":
        return _bash_is_auto_approvable(tool_input)
    if tool_name in ("Edit", "Write", "MultiEdit"):
        return _edit_is_auto_approvable(tool_input, cwd)
    return False
