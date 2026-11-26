# Copyright 2026 TensorPlay Authors
# SPDX-License-Identifier: Apache-2.0
"""DoRA: magnitude/direction decomposition, zero-init and export contract."""

import json

import tensorplay as tp
import tensorplay.nn as nn

from megatrain.lora.dora import DoRAConfig, DoRALinear, inject_dora


class TinyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(4, 6, bias=False, device="cpu", dtype=tp.float32)

    def forward(self, x):
        return self.q_proj(x)


def _config():
    return DoRAConfig(r=4, lora_alpha=8, target_modules=("q_proj",))


def test_dora_zero_init_matches_base():
    tp.manual_seed(31)
    model = TinyPolicy()
    x = tp.ones((2, 4), dtype=tp.float32)
    with tp.no_grad():
        baseline = model(x).detach().clone()
    state = inject_dora(model, _config())
    assert isinstance(model.q_proj, DoRALinear)
    assert set(state.layers) == {"q_proj"}
    with tp.no_grad():
        assert float((model(x) - baseline).abs().sum()) < 1.0e-5


def test_dora_trains_magnitude_and_factors():
    tp.manual_seed(32)
    model = TinyPolicy()
    state = inject_dora(model, _config())
    trainable = sum(int(p.numel()) for p in model.parameters() if p.requires_grad)
    assert trainable == state.parameter_count
    out = model(tp.ones((2, 4), dtype=tp.float32))
    out.sum().backward()
    layer = state.layers["q_proj"]
    assert layer.magnitude.grad is not None
    assert layer.lora_A.grad is not None
    assert layer.lora_B.grad is not None


def test_dora_export_manifest_and_magnitude_tensor(tmp_path):
    tp.manual_seed(33)
    model = TinyPolicy()
    state = inject_dora(model, _config())
    state.export(tmp_path / "adapter")
    manifest = json.loads((tmp_path / "adapter" / "adapter_config.json").read_text())
    assert manifest["use_dora"] is True
    weights = tp.load(str(tmp_path / "adapter" / "adapter_model.safetensors"))
    key = "q_proj.lora_magnitude_vector.default.weight"
    assert key in weights
    assert tuple(weights[key].shape) == (6,)


def test_dora_rslora_scaling():
    assert DoRAConfig(r=16, lora_alpha=32, use_rslora=True).scaling == 8.0
    assert DoRAConfig(r=16, lora_alpha=32).scaling == 2.0
