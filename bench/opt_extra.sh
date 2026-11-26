#!/bin/bash
# Copyright 2026 TensorPlay Authors
# SPDX-License-Identifier: Apache-2.0

# Two follow-up probes behind bench/results/opt_accum_precision.md:
#   1. equal effective batch 16: accum(4x4) vs Megatron batch 16
#   2. precision accounting: autocast fp32-CE vs no-autocast(bf16-CE) vs no-autocast+fp32-CE
#
# Idle-GPU only, same guard as opt_sweep.sh (this box is shared).
set -u
cd "$(dirname "$0")/.." || exit 1
SCRATCH=${SCRATCH:-$(mktemp -d)}
mkdir -p "$SCRATCH"
SP=${SP:-$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')/nvidia}
export LD_LIBRARY_PATH=$SP/nccl/lib:$SP/cu13/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=bench
MIN_FREE=${MIN_FREE:-18000}
REPS=${REPS:-3}

wait_idle() {
  for _ in $(seq 180); do
    if [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)" -ge "$MIN_FREE" ] \
       && [ "$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')" -eq 0 ]; then
      sleep 2
      [ "$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')" -eq 0 ] && return 0
    fi
    sleep 5
  done
  echo "GPU stayed busy; aborting" >&2
  exit 1
}

fmt() {
  python -c "import json,sys; d=json.load(sys.stdin); print(f\"{d['tokens_per_second']:>8,.0f}\t{d['step_ms']:>7.2f}\t{d['peak_memory_mib']:>7.0f}\tloss={d['loss']}\")"
}

tp() {
  wait_idle
  PYTHONPATH=bench python bench/tp_gpt_bench.py --seq 1024 --hidden 1024 --layers 2 \
    --heads 8 --vocab 65536 --precision bf16 --attn sdpa --seed 0 "$@" 2>/dev/null | fmt || echo FAILED
}

mg() {
  wait_idle
  PYTHONPATH=bench python bench/megatron_bench.py --seq 1024 --hidden 1024 --layers 2 \
    --heads 8 --vocab 65536 --precision bf16 --te --fused-adam "$@" 2>/dev/null | fmt || echo FAILED
}

run_n() {  # run_n <label> <function> <args...>
  local label="$1" fn="$2"; shift 2
  echo "### $label"
  for _ in $(seq "$REPS"); do "$fn" "$@"; done
}

{
  echo "# 1. equal effective batch 16"
  run_n "TP accum 4 x batch 4 + compile" tp --batch 4 --accum 4 --steps 6 --compile
  run_n "MG batch 16" mg --batch 16 --steps 6

  echo "# 2. precision accounting (batch 4, 10 steps, seed 0)"
  run_n "TP compile, autocast (fp32 LN/CE)" tp --batch 4 --steps 10 --compile
  run_n "TP compile, no-autocast + fp32 CE" tp --batch 4 --steps 10 --compile --no-autocast --ce-fp32
  run_n "TP compile, no-autocast (bf16 LN/CE)" tp --batch 4 --steps 10 --compile --no-autocast
} 2>&1 | tee "${OUT:-$SCRATCH/opt_extra.log}"
