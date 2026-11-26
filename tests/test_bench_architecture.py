# Copyright 2026 TensorPlay Authors
# SPDX-License-Identifier: Apache-2.0
"""Guard: the TensorPlay GPT benchmark really is Megatron-Core's GPT geometry.

The hand-rolled model in ``bench/tp_gpt_bench.py`` must have exactly the
parameter count of an untied Megatron GPT with the same config, otherwise the
head-to-head numbers compare different models. Megatron-Core reports 160.46M
parameters for (vocab=65536, hidden=1024, layers=2, seq=1024); the formula
below must reproduce both that and the benchmark model at tiny scale.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))

tp = pytest.importorskip("tensorplay")
from tp_gpt_bench import GPT  # noqa: E402


def analytic_params(vocab: int, seq: int, hidden: int, layers: int) -> int:
    """Pre-norm GPT: learned abs. pos., 4x GELU MLP, biased linears, untied head."""
    embedding = vocab * hidden + seq * hidden
    per_layer = 12 * hidden**2 + 13 * hidden
    return embedding + layers * per_layer + 2 * hidden + hidden * vocab


@pytest.mark.parametrize(
    "vocab,seq,hidden,layers,heads",
    [(512, 64, 32, 2, 4), (1024, 128, 64, 1, 8), (4096, 256, 128, 3, 4)],
)
def test_model_matches_analytic_gpt_param_count(vocab, seq, hidden, layers, heads):
    model = GPT(vocab, seq, hidden, layers, heads, attn="math")
    actual = sum(p.numel() for p in model.parameters())
    assert actual == analytic_params(vocab, seq, hidden, layers)


def test_reference_point_matches_megatron_160_46m():
    # Megatron-Core reported params_m = 160.46 for this exact configuration.
    total = analytic_params(vocab=65536, seq=1024, hidden=1024, layers=2)
    assert round(total / 1e6, 2) == 160.46
    _ = tp
