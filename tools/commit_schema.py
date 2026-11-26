#!/usr/bin/env python3
# Copyright 2026 TensorPlay Authors
# SPDX-License-Identifier: Apache-2.0
"""Validate commit subjects against this repository's convention.

The format is Conventional Commits with one project-specific type, ``bench``,
for changes to the benchmark harness or to recorded benchmark evidence::

    type(scope): subject

where ``type`` is one of the values below, ``scope`` is an optional lowercase
area (``trainer``, ``losses``, ``bench``, ``tools``, ...) and ``subject`` is a
one-line summary in the imperative mood that does not end with a period.

Usage::

    python tools/commit_schema.py --message "feat(trainer): add a hook"
    python tools/commit_schema.py --commit-msg-file .git/COMMIT_EDITMSG
    python tools/commit_schema.py --range origin/main..HEAD --require-signoff

Exit status is 1 when any message in the input violates the schema.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys

TYPES = (
    "bench",
    "build",
    "chore",
    "ci",
    "docs",
    "feat",
    "fix",
    "perf",
    "refactor",
    "revert",
    "style",
    "test",
)
SCOPE_RE = re.compile(r"^[a-z][a-z0-9-]{0,23}$")
HEADER_RE = re.compile(
    r"^(?P<type>[a-z]+)(?:\((?P<scope>[^()]*)\))?(?P<breaking>!)?: (?P<subject>\S.*)$"
)
BREAKING_RE = re.compile(r"^BREAKING[ -]CHANGE: \S.*$", re.MULTILINE)
SIGNED_OFF_RE = re.compile(r"^Signed-off-by: .+ <[^<>@\s]+@[^<>@\s]+>$")
SUGGESTION = (
    "  expected: type(scope): subject\n"
    f"  types: {', '.join(TYPES)}\n"
    "  scope: optional lowercase area, e.g. trainer, losses, bench, tools\n"
    "  subject: imperative mood, no trailing period, header <= 100 chars"
)
HEADER_MAX = 100
#: Subjects git itself writes; they are exempt like merge and revert commits.
EXEMPT_PREFIXES = ("Merge ", "Revert ", "fixup!", "squash!", "amend!")


def validate_header(header: str) -> list[str]:
    """Return the problems with one subject line (empty when it is valid)."""
    if not header.strip():
        return ["empty commit message", SUGGESTION]
    if header.startswith(EXEMPT_PREFIXES):
        return []
    match = HEADER_RE.match(header)
    if not match:
        return [f"header does not match 'type(scope): subject': {header!r}", SUGGESTION]

    problems: list[str] = []
    commit_type = match.group("type")
    scope = match.group("scope")
    subject = match.group("subject")
    if commit_type not in TYPES:
        problems.append(f"unknown type {commit_type!r}")
    if scope is not None and not SCOPE_RE.match(scope):
        problems.append(f"scope {scope!r} must be lowercase letters, digits or dashes")
    if len(header) > HEADER_MAX:
        problems.append(f"header is {len(header)} characters, limit is {HEADER_MAX}")
    if subject.rstrip().endswith("."):
        problems.append("subject must not end with a period")
    if problems:
        problems.append(SUGGESTION)
    return problems


def validate_message(message: str, require_signoff: bool = False) -> list[str]:
    """Return the problems with a full commit message body."""
    lines = [line for line in message.splitlines() if not line.startswith("#")]
    header = next((line.strip() for line in lines if line.strip()), "")
    if header.startswith(EXEMPT_PREFIXES):
        return []
    problems = validate_header(header)
    if problems:
        return problems
    match = HEADER_RE.match(header)
    if match is not None and match.group("breaking") and not BREAKING_RE.search(message):
        problems.append("breaking change needs a 'BREAKING CHANGE: <what breaks>' footer")
    if require_signoff and not any(
        SIGNED_OFF_RE.match(line.strip()) for line in message.splitlines()
    ):
        problems.append(
            "missing 'Signed-off-by: Name <email>' trailer; commit with 'git commit -s'"
        )
    return problems


def commits_in_range(rev_range: str) -> list[tuple[str, str]]:
    """Return ``(sha, message)`` for every commit in ``rev_range``."""
    result = subprocess.run(
        ["git", "log", "--format=%H%x1f%B%x1e", rev_range],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise SystemExit(f"git log failed for {rev_range!r}: {result.stderr.strip()}")
    entries = []
    for chunk in result.stdout.split("\x1e"):
        chunk = chunk.strip("\n")
        if not chunk:
            continue
        sha, _, message = chunk.partition("\x1f")
        entries.append((sha, message))
    return entries


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--message", help="validate this commit message")
    source.add_argument("--commit-msg-file", help="validate this file (the commit-msg hook's path)")
    source.add_argument("--range", help="validate every commit in this revision range")
    parser.add_argument(
        "--require-signoff",
        action="store_true",
        help="also require a Signed-off-by trailer (needs --range or a message)",
    )
    args = parser.parse_args()

    failures: list[tuple[str, list[str]]] = []
    if args.range:
        for sha, message in commits_in_range(args.range):
            problems = validate_message(message, args.require_signoff)
            if problems:
                subject = next((line for line in message.splitlines() if line.strip()), "")
                failures.append((f"{sha[:8]} {subject}", problems))
    else:
        if args.commit_msg_file:
            with open(args.commit_msg_file, encoding="utf-8") as handle:
                message = handle.read()
        else:
            message = args.message
        problems = validate_message(message, args.require_signoff)
        if problems:
            failures.append(("commit message", problems))

    for where, problems in failures:
        print(f"commit schema: FAIL -> {where}", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
    if failures:
        return 1
    print("commit schema: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
