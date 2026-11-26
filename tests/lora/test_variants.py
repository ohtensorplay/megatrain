# Copyright 2026 TensorPlay Authors
# SPDX-License-Identifier: Apache-2.0
"""LoRA variants copied from the local reference stacks: rsLoRA and LoRA+."""

import json
import math

import tensorplay as tp
import tensorplay.nn as nn

from megatrain.lora.base import LoraConfig, LoRAError, inject_lora, load_lora
from megatrain.lora.variants import lora_plus_optimizer, rslora_scaling


class TinyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(4, 6, bias=False, device="cpu", dtype=tp.float32)

    def forward(self, x):
        return self.q_proj(x)


def test_rslora_uses_alpha_over_sqrt_rank():
    config = LoraConfig(r=16, lora_alpha=32, use_rslora=True)
    assert config.scaling == 32 / math.sqrt(16)
    assert rslora_scaling(config) == config.scaling
    assert LoraConfig(r=16, lora_alpha=32).scaling == 2.0


def test_rslora_flag_survives_adapter_roundtrip(tmp_path):
    tp.manual_seed(21)
    model = TinyPolicy()
    state = inject_lora(model, LoraConfig(r=4, lora_alpha=8, use_rslora=True))
    state.export(tmp_path / "adapter")
    manifest = json.loads((tmp_path / "adapter" / "adapter_config.json").read_text())
    assert manifest["use_rslora"] is True

    fresh = TinyPolicy()
    restored = load_lora(fresh, tmp_path / "adapter")
    assert restored.config.use_rslora is True
    assert restored.config.scaling == 8 / math.sqrt(4)


def test_lora_plus_optimizer_sets_two_learning_rates():
    tp.manual_seed(22)
    model = TinyPolicy()
    state = inject_lora(model, LoraConfig(r=4, lora_alpha=8))
    optimizer = lora_plus_optimizer(state, lr=1.0e-4, ratio=16.0)
    assert len(optimizer.param_groups) == 2
    assert optimizer.param_groups[0]["lr"] == 1.0e-4
    assert optimizer.param_groups[1]["lr"] == 16.0e-4
    a_ids = {id(p) for p in optimizer.param_groups[0]["params"]}
    b_ids = {id(p) for p in optimizer.param_groups[1]["params"]}
    assert id(state.layers["q_proj"].lora_A) in a_ids
    assert id(state.layers["q_proj"].lora_B) in b_ids
    assert not a_ids & b_ids


def test_lora_plus_ratio_must_be_positive():
    model = TinyPolicy()
    state = inject_lora(model, LoraConfig(r=4, lora_alpha=8))
    try:
        lora_plus_optimizer(state, lr=1.0e-4, ratio=0.0)
    except LoRAError:
        pass
    else:
        raise AssertionError("non-positive LoRA+ ratio must be rejected")
