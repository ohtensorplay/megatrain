## What and why

<!-- One or two sentences: the problem this solves, the change, and what it is not. -->

## Evidence

<!--
Anything that can move a throughput or memory number needs the exact command,
the quiet-window guard output, and the before/after figures. Docs-only changes
say so in one sentence. Numbers without a command are not evidence.
-->

## Checklist

- [ ] I, the PR author, have personally reviewed every line of this PR.
- [ ] `PYTHONPATH=src python -m pytest tests -q` passes (GPU cases skip without a GPU).
- [ ] `make check` passes: ruff check, ruff format --check, license headers.
- [ ] Every new number in `README.md` or `devdocs/` carries the command that produced it.
- [ ] Benchmark evidence under `bench/results/` is updated, not only quoted in prose.
- [ ] Copied or adapted third-party code is attributed in the file header and the `NOTICE`.