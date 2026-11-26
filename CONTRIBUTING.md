# Contributing to megaTrain

megaTrain is the training core of the TensorPlay stack. It owns *how* a step
runs — optimizer, FSDP, compile, AMP, gradient accumulation, clipping,
checkpointing — and keeps *what* is being trained in the objective that the
caller injects. Anything that knows about rollouts, rewards or policy versions
belongs in `megaRL`, not here.

## Development setup

Python 3.12 or newer, and a TensorPlay build from its own index:

```bash
make install        # editable install with the dev extras
make hooks          # pre-commit and commit-msg hooks
```

`make install` is

```bash
python -m pip install -e ".[dev]" \
    --extra-index-url https://download.tensorplay.cn/simple
```

On a machine where `import tensorplay` fails with a missing `libnccl.so.2`, put
the NCCL libraries shipped with the wheel on the loader path:

```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.13/site-packages/nvidia/nccl/lib:$LD_LIBRARY_PATH
```

Tests do not need that path; only a real training run does.

## Quality gates

```bash
make check          # lint + format check + license headers + tests
```

is exactly what CI runs, so a red local run and a red pull request mean the
same thing. `make format` applies the automatic fixes.

Two rules that are not negotiable in this repository:

- **No stubs or fake tensors.** CPU tests use tiny *real* TensorPlay modules
  with real optimizers; a path that cannot run on CPU gets a GPU test that
  skips when no device is present, never a mock that hides a broken kernel.
- **Every numerics claim carries a test.** If a commit says two losses agree,
  a test compares them, including `ignore_index` rows and non-block-aligned
  vocabulary sizes.

## Benchmarks are evidence, not decoration

A throughput or memory number is only accepted with the command that produced
it and the window it was produced in. Concretely:

1. **Same structure, same dtype, same token budget, same optimizer.** A
   configuration change is a different experiment and says so.
2. **Only the installed vendor stack counts as a baseline.** Megatron-Core
   numbers measured on its uninstalled torch fallback are invalid; the same is
   true of any pre-Transformer-Engine figure. `bench/results/te_install.md`
   records how the valid baseline was established.
3. **No precision shortcuts in a headline.** Megatron computes its loss in
   fp32, so a bf16-CE speedup is reported as an option, never as a comparable
   win.
4. **Wait for a quiet card.** `make bench-guard` prints whether the GPU is
   usable; the harnesses require `utilization = 0` and a free-memory floor
   (`MIN_FREE_MIB`, 8000 by default). A rep from a disturbed window is
   discarded and retried, not averaged in.
5. **Write it down.** Results belong in `bench/results/*.md` with the command
   and the guard output, and the prose in `devdocs/` quotes them. A number that
   exists only in a chat message does not survive the next measurement.
6. **Cross-stack comparisons come from one window** when possible; when they
   cannot, the document says which side is from an older window and why.

## Commits

The subject line follows Conventional Commits:

```
type(scope): subject
```

Types: `bench`, `build`, `chore`, `ci`, `docs`, `feat`, `fix`, `perf`,
`refactor`, `revert`, `style`, `test`. `bench` is local to this repository and
covers the harness and its recorded evidence. The scope is an optional
lowercase area (`trainer`, `losses`, `batch`, `kl`, `tools`, `repo`, ...).

- One logical change per commit; a formatting pass is its own commit.
- The subject is imperative, has no trailing period, and stays within 100
  characters.
- **Sign your commits** (`git commit -s`). The `Signed-off-by` trailer is
  checked by the commit-msg hook and by CI; it is the Developer Certificate of
  Origin, not a copyright assignment.
- Breaking changes carry a `BREAKING CHANGE:` footer.

`python tools/commit_schema.py --message "..."` validates a message without
committing.

## Pull requests

The template is enforced: `## What and why`, `## Evidence`, `## Checklist`,
and the self-review box. A change that can move a number needs the command and
the figures in `## Evidence`; a documentation-only change says so in one
sentence. Reviewers look for the same thing the tests look for — a claim, and
the thing that proves it.

A squash merge takes the pull-request title as its subject, so the title must
satisfy the commit schema too.

## Code provenance

Do not copy code from another repository into this one. When a piece of
third-party code is adapted and there is no alternative, say so in the pull
request and in the file header: source URL, revision, files, license, and where
the notice is retained. `NOTICE` lists what is bundled and what is only
measured as an external baseline.

## Public tree hygiene

A machine-specific absolute path must never land in a tracked file: not a
conda install's `site-packages`, not one user's scratch directory, and not
the name or location of a private upstream checkout. Derive paths at
runtime (`sysconfig.get_paths()`, `mktemp`) or accept them through `SP`,
`OUT` and `SCRATCH` environment variables.

Machine-specific notes belong in `devdocs/local/`, which is ignored by git
(the same pattern Megatron-LM uses with its ignored `local/` directory).
`make lint` runs `tools/check_public.py` over the tracked files to enforce
both rules; CI runs it too.

## Reporting problems

Open a bug report with the exact command, the GPU and versions
(`python -c "import torch, tensorplay; print(torch.__version__, tensorplay.__version__)"`),
and whether the card was quiet. Security issues go through
[`SECURITY.md`](SECURITY.md), not the public tracker.
