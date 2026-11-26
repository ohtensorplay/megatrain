#!/bin/bash
# Copyright 2026 TensorPlay Authors
# SPDX-License-Identifier: Apache-2.0

# Same-window comparison: Megatron-Core + TE against TensorPlay with and without
# the fused vocabulary loss. Only a quiet GPU counts; a disturbed run is
# discarded and retried, so the bundle is from one consistent window.
set -u
cd "$(dirname "$0")/.." || exit 1
: > /tmp/opencode/mg_vs_fused.log
for attempt in $(seq 1 8); do
  MG=1 AMP=0 GEOM=160m REPS=3 STEPS=12 MIN_FREE=${MIN_FREE:-8000} \
    OUT=/tmp/opencode/mg_vs_fused_run.log ./bench/fused_ce.sh \
    >> /tmp/opencode/mg_vs_fused.log 2>&1
  rows=$(grep -cE '^\s+[0-9,]+' /tmp/opencode/mg_vs_fused.log)
  echo "attempt $attempt: $rows rows" >> /tmp/opencode/mg_vs_fused.log
  # 7 configs x 3 reps = 21 clean rows
  [ "$rows" -ge 21 ] && break
  sleep 180
done
