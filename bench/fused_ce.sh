#!/bin/bash
# Copyright 2026 TensorPlay Authors
# SPDX-License-Identifier: Apache-2.0

# Fused vocabulary projection + CE: A/B against the same stack without it.
#
# Same geometry, same dtype, same token budget, same optimizer; the only
# difference is whether the loss keeps its own logits buffers
# (F.cross_entropy) or reuses the projection's buffer (F.linear_cross_entropy).
#
# The box is shared with another session, so every rep waits for an idle GPU
# and a disturbed window is thrown away and retried.
set -u
cd "$(dirname "$0")/.." || exit 1
SCRATCH=${SCRATCH:-$(mktemp -d)}
mkdir -p "$SCRATCH"
SP=${SP:-$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')/nvidia}
export LD_LIBRARY_PATH=$SP/nccl/lib:$SP/cu13/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=bench

STEPS=${STEPS:-12}
REPS=${REPS:-3}
MIN_FREE=${MIN_FREE:-10000}
GEOM=${GEOM:-160m}
# AMP=1 (default) keeps fp32 weights under a bf16 autocast (the "AMP" row);
# AMP=0 casts the weights to bf16 (the row §4's head-to-head uses).
AMP=${AMP:-1}
OUT=${OUT:-$SCRATCH/fused_ce_${GEOM}.log}

case "$GEOM" in
  160m) SHAPE="--seq 1024 --batch 4 --hidden 1024 --layers 2 --heads 8 --vocab 65536" ;;
  474m) SHAPE="--seq 2048 --batch 2 --hidden 2048 --layers 4 --heads 16 --vocab 65536" ;;
  *)    echo "GEOM must be 160m or 474m" >&2; exit 2 ;;
esac

free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits; }
util_pct() { nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' '; }

wait_idle() {
  local tries=${IDLE_TRIES:-180}
  for ((i = 0; i < tries; i++)); do
    if [ "$(free_mib)" -ge "$MIN_FREE" ] && [ "$(util_pct)" -eq 0 ]; then
      sleep 2
      if [ "$(util_pct)" -eq 0 ]; then return 0; fi
    fi
    sleep 5
  done
  return 1
}

one_rep() {
  local cfg="$1" free_before free_after bench=tp_gpt_bench.py flags
  # Per-invocation scratch file: two copies of this script must not overwrite
  # each other's measurement, and a stale file must not be read as a result.
  local out_json=$SCRATCH/fused_rep.$$.json
  case "$cfg" in
    mg:*) bench=megatron_bench.py; flags="${cfg#mg:}" ;;
    *)    flags="--precision bf16 --attn sdpa $cfg"
          [ "$AMP" = "1" ] && flags="--precision bf16 --amp --attn sdpa $cfg" ;;
  esac
  # Never let a busy window look like a measurement: an empty or missing JSON
  # must be discarded and retried, not averaged in.
  if ! wait_idle; then echo BUSY; return 0; fi
  free_before=$(free_mib)
  # shellcheck disable=SC2086
  rm -f "$out_json"
  PYTHONPATH=bench python bench/$bench --steps "$STEPS" $SHAPE $flags 2>/dev/null \
    > "$out_json"
  free_after=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
  # Decide on the result while the file still exists, then remove it on every
  # path: a leftover scratch file is exactly what a later rep must not read.
  row=""
  if [ -s "$out_json" ]; then
    row=$(python -c "import json,sys; d=json.load(open(sys.argv[1])); print(f\"{d['tokens_per_second']:>8,.0f}\t{d['step_ms']:>7.2f}\t{d['peak_memory_mib']:>7.0f}\t{d['total_peak_mib']:>7.0f}\t{d['loss']}\")" "$out_json" 2>/dev/null)
  fi
  rm -f "$out_json"
  if [ -z "$row" ]; then echo "FAILED"; return; fi
  if [ $((free_before - free_after)) -gt 2048 ]; then echo "BUSY"; return; fi
  echo "$row"
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
  echo "geometry $GEOM $SHAPE steps=$STEPS reps=$REPS amp=$AMP min_free=${MIN_FREE}MiB"
  echo "#   tok/s    step ms  step MiB  total MiB  loss"
  # (1) whole-step compile, reference CE -- the previous headline knob set
  # (2) fully eager, reference CE        -- same structure as (3)
  # (3) fully eager, fused CE            -- the A/B of the loss itself
  # (4)(5) compiled trunk with the loss applied outside it, as Megatron does
  cases="--compile --fused-adam|--fused-adam|--fused-adam --fused-ce|--compile --fused-adam --eager-ce|--compile --fused-adam --eager-ce --fused-ce"
  if [ "${MG:-0}" = "1" ]; then
    cases="mg:--precision bf16 --te|mg:--precision bf16 --te --fused-adam|$cases"
  fi
  # Split on '|' through a throwaway array: setting IFS globally here would
  # also apply inside one_rep and stop $SHAPE / $flags from word-splitting.
  cfgs=()
  IFS='|' read -r -a cfgs <<< "$cases"
  for cfg in "${cfgs[@]}"; do
    echo "### ${cfg}"
    run "$cfg"
  done
} 2>&1 | tee "$OUT"
