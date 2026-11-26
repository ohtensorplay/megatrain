# Security policy

## Supported versions

Development happens on `main`; only the latest commit is supported. There are
no released versions yet, and no backports.

## Reporting a vulnerability

Report privately through
[GitHub security advisories](https://github.com/ohtensorplay/megatrain/security/advisories/new).
Do not open a public issue, and do not attach checkpoints, tokens or model
weights that you cannot share.

Please include the affected commit, the command that triggers the problem, and
the environment (`python -c "import torch, tensorplay; print(torch.__version__, tensorplay.__version__)"`
plus GPU and driver).

## What is in scope

megaTrain is a training library. In scope:

- arbitrary code execution, path traversal or symlink following while writing
  or loading checkpoints (`TrainerConfig.checkpoint_dir`, artifact sidecars);
- deserialization of untrusted checkpoints or state dictionaries;
- integrity failures in the SHA-256 manifest or the atomic `LATEST` publish;
- denial of service through unbounded memory use in a documented code path.

## What is not in scope

- Model behaviour, hallucination, or the safety of a trained model.
- Prompt injection in data; the trainer does not interpret the tokens it feeds.
- Vulnerabilities in TensorPlay, PyTorch or CUDA: report those upstream. If the
  issue is only reachable through megaTrain, tell us so we can mitigate here.
- Benchmark numbers on a shared GPU; see [`CONTRIBUTING.md`](CONTRIBUTING.md)
  for the quiet-window rules instead.