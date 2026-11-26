# Developer entry points. CI runs the same commands, so `make check` locally and
# a green pull request mean the same thing.
PY ?= python
export PYTHONPATH := src

# The benchmarks need a quiet card; this is the floor the harnesses enforce.
MIN_FREE_MIB ?= 8000

.PHONY: help install hooks lint format test check ci bench-guard

help:  ## Show this help
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) \
		| sed -E 's/^([a-z-]+):.*## /\1: /' | sed 's/^/  make /'

install:  ## Editable install with the dev extras (TensorPlay comes from its own index)
	$(PY) -m pip install -e ".[dev]" \
		--extra-index-url https://download.tensorplay.cn/simple

hooks:  ## Install the pre-commit and commit-msg hooks
	$(PY) -m pre_commit install --install-hooks
	$(PY) -m pre_commit install --hook-type commit-msg

lint:  ## Lint, formatting and license-header checks
	$(PY) -m ruff check src tests bench tools
	$(PY) -m ruff format --check src tests bench tools
	$(PY) tools/check_headers.py

format:  ## Apply the automatic fixes
	$(PY) -m ruff check --fix src tests bench tools
	$(PY) -m ruff format src tests bench tools

test:  ## Run the test suite (GPU cases skip without a GPU)
	$(PY) -m pytest tests -q

check: lint test  ## Everything CI runs

ci: check  ## Alias for check

bench-guard:  ## Show whether the GPU is quiet enough to time a benchmark
	@nvidia-smi --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits \
		| awk -F', ' 'NF == 2 { free = $$1; util = $$2; \
		if (free < $(MIN_FREE_MIB)) { printf "busy: %d MiB free, need $(MIN_FREE_MIB)\n", free; exit 1 } \
		if (util != 0) { printf "busy: util %d%%\n", util; exit 1 } \
		printf "quiet: %d MiB free, util 0%%\n", free }'