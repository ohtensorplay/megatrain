#!/usr/bin/env python3
# Copyright 2026 TensorPlay Authors
# SPDX-License-Identifier: Apache-2.0
"""Keep the published tree free of machine-specific paths and private names.

With no arguments, scans every tracked file. With explicit paths, scans just
those (which is how the pre-commit hook calls it)::

    python tools/check_public.py
    python tools/check_public.py README.md bench/fused_ce.sh

Machine-specific notes belong in the ignored ``devdocs/local/`` directory.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

# The private checkout directory's name is assembled from fragments so this
# gate never has to print it in the public repository itself.
_PRIVATE = "re" + "fs"
PRIVATE_NAME = re.compile(r"(?<![A-Za-z0-9_])" + _PRIVATE + r"(?![A-Za-z0-9_])", re.I)


def _absolute(*parts: str) -> str:
    return "/" + "/".join(parts) + "/"


FORBIDDEN_PATHS = (
    _absolute("home"),
    _absolute("root"),
    _absolute("Users"),
    _absolute("mnt"),
    _absolute("media"),
    _absolute("var"),
    _absolute("tmp"),
    "/" + "usr" + "/" + "local" + "/",
    "C:" + "\\",
    "C:" + "/",
)
ABSOLUTE = re.compile("|".join(re.escape(item) for item in FORBIDDEN_PATHS))


def tracked_files() -> list[Path]:
    result = subprocess.run(["git", "ls-files", "-z"], capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"git ls-files failed: {result.stderr.strip()}")
    return [Path(name) for name in result.stdout.split("\0") if name]


def check(paths: list[Path]) -> list[str]:
    problems: list[str] = []
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for number, line in enumerate(text.splitlines(), 1):
            if ABSOLUTE.search(line):
                problems.append(f"{path}:{number}: machine-specific absolute path")
            if PRIVATE_NAME.search(line):
                problems.append(f"{path}:{number}: private checkout name")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="check these files instead of the tracked tree",
    )
    args = parser.parse_args()
    problems = check(args.paths if args.paths else tracked_files())
    for problem in problems:
        print(problem, file=sys.stderr)
    if problems:
        print(
            f"public tree: {len(problems)} problem(s); local notes go in devdocs/local/",
            file=sys.stderr,
        )
        return 1
    print("public tree: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
