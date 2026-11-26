#!/usr/bin/env python3
# Copyright 2026 TensorPlay Authors
# SPDX-License-Identifier: Apache-2.0
"""Check a pull-request description against the repository template.

This project's whole point is measured training throughput, so a pull request
that could change a number has to say which numbers it moved and how they were
measured. The template encodes that as required sections; this script keeps
them from being silently dropped.

Reads the description from stdin (or ``PR_BODY``) and prints one line per
problem. Exit status is 1 when the description is incomplete.

Usage::

    python tools/check_pr_body.py < body.md
    PR_BODY="$(gh pr view 12 --json body -q .body)" python tools/check_pr_body.py
"""

from __future__ import annotations

import os
import re
import sys

#: Must match the checkboxes and headings in .github/PULL_REQUEST_TEMPLATE.md.
REVIEW_BOX = "I, the PR author, have personally reviewed every line of this PR."
REQUIRED_HEADINGS = ("What and why", "Evidence", "Checklist")
HEADING_RE = re.compile(r"^##[ \t]+(?P<title>\S.*?)[ \t]*$", re.MULTILINE)
FENCE_RE = re.compile(r"^[ \t]*(`{3,}|~{3,}).*?(?:^[ \t]*\1[^\n]*$|\Z)", re.MULTILINE | re.DOTALL)
COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
CHECK_RE = re.compile(
    r"^[ \t]*[-*][ \t]+\[(?P<mark>[xX ])\][ \t]+"
    + re.escape(REVIEW_BOX.rstrip("."))
    + r"\.?[ \t]*$",
    re.MULTILINE,
)
#: A section is a placeholder when its only content is one of these.
PLACEHOLDER_RE = re.compile(r"^(?:n/?a|none|todo|tbd|tba|fill (?:in|me)|[.?…-]+)$", re.IGNORECASE)
#: Ten-ish characters is enough to force a sentence; a single word is not.
MIN_SECTION_CHARS = 12


def scrub(body: str) -> str:
    """Remove fenced code and HTML comments before reading the structure.

    Neither may tick the review box or supply a heading, but both stay in the
    description for humans and are printed below by the author's own choice.
    """
    return COMMENT_RE.sub("", FENCE_RE.sub("", body))


def sections(body: str) -> dict[str, str]:
    """Map heading title (lowercased) to the text that follows it."""
    found: dict[str, str] = {}
    matches = list(HEADING_RE.finditer(body))
    for index, match in enumerate(matches):
        stop = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        found[match.group("title").lower()] = body[match.end() : stop]
    return found


def check(body: str) -> list[str]:
    """Return the problems with a pull-request description."""
    problems: list[str] = []
    text = scrub(body)

    matches = list(CHECK_RE.finditer(text))
    if not matches:
        problems.append(f"missing the self-review checkbox: '[x] {REVIEW_BOX}'")
    elif not any(match.group("mark").lower() == "x" for match in matches):
        problems.append("the self-review checkbox is present but not ticked")

    found = sections(text)
    for title in REQUIRED_HEADINGS:
        if title.lower() not in found:
            problems.append(f"missing the '## {title}' section")
    if problems:
        return problems

    evidence = " ".join(found["evidence"].split())
    if PLACEHOLDER_RE.match(evidence.strip("*.` ")):
        problems.append("'## Evidence' is a placeholder; state what was measured")
    elif len(evidence) < MIN_SECTION_CHARS:
        problems.append(
            "'## Evidence' is empty: give the command and the numbers, or explain "
            "why this change cannot move one"
        )
    return problems


def main() -> int:
    body = os.environ.get("PR_BODY")
    if body is None:
        body = sys.stdin.read()
    problems = check(body)
    if problems:
        print("pull-request description: FAIL", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        print("  template: .github/PULL_REQUEST_TEMPLATE.md", file=sys.stderr)
        return 1
    print("pull-request description: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
