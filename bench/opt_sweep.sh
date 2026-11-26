#!/bin/bash
# Copyright 2026 TensorPlay Authors
# SPDX-License-Identifier: Apache-2.0

# Extreme-optimization sweep for the 160.46M GPT on one RTX 3090.
#
# The box is shared: another session runs vLLM in bursts, which can halve
# throughput. Every measurement therefore waits for an idle GPU (util 0,
# >= MIN_FREE MiB free), and any rep whose window was disturbed is discarded
# and retried, so the logged medians come from quiet windows only.
set -u
cd "$(dirname "$0")/.." || exit 1
SCRATCH=${SCRATCH:-$(mktemp -d)}
mkdir -p "$SCRATCH"
SP=${SP:-$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')/nvidia}
export LD_LIBRARY_PATH=$SP/nccl/lib:$SP/cu13/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=bench

BATCH=${BATCH:-4}
STEPS=${STEPS:-10}
REPS=${REPS:-3}
MIN_FREE=${MIN_FREE:-18000}
OUT=${OUT:-$SCRATCH/opt_sweep.log}

free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits; }
util_pct() { nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' '; }

wait_idle() {
  for _ in $(seq 180); do
    if [ "$(free_mib)" -ge "$MIN_FREE" ] && [ "$(util_pct)" -eq 0 ]; then
      sleep 2
      if [ "$(util_pct)" -eq 0 ]; then return 0; fi
    fi
    sleep 5
  done
  echo "GPU stayed busy for 15 min; aborting" >&2
  exit 1
}

# Emits "<tok/s>\t<step ms>\t<peak MiB>" for one rep, or "BUSY" if the window
# was disturbed (co-tenant touched the GPU while we were timing).
one_rep() {
  local cfg="$1" free_before busy_mid
  wait_idle
  free_before=$(free_mib)
  # shellcheck disable=SC2086
  PYTHONPATH=bench python bench/tp_gpt_bench.py \
    --steps "$STEPS" --seq 1024 --batch "$BATCH" --hidden 1024 --layers 2 \
    --heads 8 --vocab 65536 --precision bf16 --attn sdpa $cfg 2>/dev/null \
    > "$SCRATCH/one_rep.json"
  busy_mid=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
  if [ ! -s "$SCRATCH/one_rep.json" ]; then echo "FAILED"; return; fi
  # >2 GiB of foreign allocation appearing mid-run means another tenant moved.
  if [ $((free_before - busy_mid)) -gt 2048 ]; then echo "BUSY"; return; fi
  python -c "import json,sys; d=json.load(open(sys.argv[1])); print(f\"{d['tokens_per_second']:>8,.0f}\t{d['step_ms']:>7.2f}\t{d['peak_memory_mib']:>7.0f}\t{d.get('total_peak_mib', 0):>7.0f}\")" "$SCRATCH/one_rep.json"
}

run() {
  local cfg="$1" i=0 good=0
  while [ "$good" -lt "$REPS" ] && [ "$i" -lt $((REPS * 3)) ]; do
    local out
    out=$(one_rep "$cfg")
    i=$((i + 1))
    case "$out" in
      BUSY) echo "  (discarded a disturbed window)" ;;
      FAILED) echo "  FAILED" ;;
      *) echo "$out"; good=$((good + 1)) ;;
    esac
  done
}

{
  echo "geometry batch=$BATCH steps=$STEPS reps=$REPS min_free=${MIN_FREE}MiB"
  for cfg in "" "--compile" "--compile-max-autotune" "--compile-fullgraph" \
             "--fused-ce --compile" "--cuda-graph --fused-adam" \
             "--fused-adam --compile" "--accum 4 --compile" "--accum 4" \
             "--no-autocast --ce-fp32 --compile" "--no-autocast --compile"; do
    echo "### ${cfg:-(baseline)}"
    echo "#   tok/s    step ms  step MiB  total MiB"
    run "$cfg"
  done
} 2>&1 | tee "$OUT"
