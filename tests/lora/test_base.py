# Copyright 2026 TensorPlay Authors
# SPDX-License-Identifier: Apache-2.0
"""TensorPlay LoRA: injection, frozen base, zero init and adapter roundtrip."""

import json

import pytest
import tensorplay as tp
import tensorplay.nn as nn

from megatrain.lora.base import LoraConfig, inject_lora, load_lora


class TinyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(4, 6, bias=False, device="cpu", dtype=tp.float32)
        self.gate_proj = nn.Linear(4, 6, bias=False, device="cpu", dtype=tp.float32)
        self.other = nn.Linear(4, 6, bias=False, device="cpu", dtype=tp.float32)

    def forward(self, x):
        return self.q_proj(x) + self.gate_proj(x) + self.other(x)


def _config():
    return LoraConfig(r=4, lora_alpha=8, target_modules=("q_proj", "gate_proj"))


def test_inject_lora_freezes_base_and_targets_only():
    tp.manual_seed(7)
    model = TinyPolicy()
    before = model.other.weight.detach().clone()
    state = inject_lora(model, _config())
    assert set(state.layers) == {"q_proj", "gate_proj"}
    assert state.parameter_count == 2 * (4 * 4 + 6 * 4)
    trainable = sum(int(p.numel()) for p in model.parameters() if p.requires_grad)
    assert trainable == state.parameter_count
    assert float((model.other.weight.detach() - before).abs().sum()) == pytest.approx(0.0)


def test_zero_init_matches_base_forward():
    tp.manual_seed(7)
    model = TinyPolicy()
    x = tp.ones((2, 4), dtype=tp.float32)
    with tp.no_grad():
        baseline = model(x).detach().clone()
    inject_lora(model, _config())
    with tp.no_grad():
        after = model(x).detach()
    assert float((after - baseline).abs().sum()) == pytest.approx(0.0)


def test_lora_delta_is_trainable(tmp_path):
    tp.manual_seed(7)
    model = TinyPolicy()
    state = inject_lora(model, _config())
    x = tp.ones((2, 4), dtype=tp.float32)
    with tp.no_grad():
        baseline = model(x).detach().clone()
        state.layers["q_proj"].lora_B.data = tp.ones((6, 4), dtype=tp.float32)
    with tp.no_grad():
        after = model(x).detach()
    assert float((after - baseline).abs().sum()) > 0


def test_export_and_load_roundtrip(tmp_path):
    tp.manual_seed(7)
    model = TinyPolicy()
    state = inject_lora(model, _config())
    with tp.no_grad():
        state.layers["q_proj"].lora_B.data = tp.ones((6, 4), dtype=tp.float32) * 0.5
    state.export(tmp_path / "adapter")
    manifest = json.loads((tmp_path / "adapter" / "adapter_config.json").read_text())
    assert manifest["peft_type"] == "LORA"
    assert manifest["r"] == 4 and manifest["lora_alpha"] == 8

    restored_model = TinyPolicy()
    restored = load_lora(restored_model, tmp_path / "adapter")
    assert set(restored.layers) == set(state.layers)
    assert restored.parameter_count == state.parameter_count
    for name in state.layers:
        original = state.layers[name]
        copy = restored.layers[name]
        assert float((copy.lora_B.data - original.lora_B.data).abs().sum()) == pytest.approx(0.0)
