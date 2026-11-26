#!/usr/bin/env python3
# Copyright 2026 TensorPlay Authors
# SPDX-License-Identifier: Apache-2.0
"""Check that every Python and shell source carries the project SPDX header.

The header is two comment lines at the top of the file (after a shebang, when
there is one)::

    # Copyright 2026 TensorPlay Authors
    # SPDX-License-Identifier: Apache-2.0

Usage::

    python tools/check_headers.py                  # scan the package and benches
    python tools/check_headers.py path ...         # check exactly these files
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

COPYRIGHT = "# Copyright 2026 TensorPlay Authors"
LICENSE = "# SPDX-License-Identifier: Apache-2.0"
SCAN_ROOTS = ("src", "tests", "bench", "tools")
SUFFIXES = (".py", ".sh")
# Lines a header may sit behind: a shebang, or a coding declaration on Python 2
# era files.  Nothing else counts, so a header buried in a docstring fails.
PREAMBLE = ("#!", "# -*- coding", "# coding:")


def _first_lines(path: Path, count: int = 5) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return []
    return text.splitlines()[:count]


def has_header(path: Path) -> bool:
    lines = _first_lines(path)
    seen = [line.strip() for line in lines if not line.strip().startswith(PREAMBLE)]
    return any(line == COPYRIGHT for line in seen) and any(
        line == LICENSE for line in seen
    )


def iter_sources(paths: list[Path]) -> list[Path]:
    if paths:
        return [path for path in paths if path.suffix in SUFFIXES]
    found: list[Path] = []
    for root in SCAN_ROOTS:
        base = Path(root)
        if not base.is_dir():
            continue
        found.extend(
            path
            for path in sorted(base.rglob("*"))
            if path.is_file() and path.suffix in SUFFIXES
        )
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path)
    args = parser.parse_args()

    missing = [path for path in iter_sources(args.paths) if not has_header(path)]
    for path in missing:
        print(f"{path}: missing the SPDX header", file=sys.stderr)
    if missing:
        print(
            f"{len(missing)} file(s) must start with:\n  {COPYRIGHT}\n  {LICENSE}",
            file=sys.stderr,
        )
        return 1
    print(f"license headers: ok ({len(iter_sources(args.paths))} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())