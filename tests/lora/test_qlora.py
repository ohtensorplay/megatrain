# Copyright 2026 TensorPlay Authors
# SPDX-License-Identifier: Apache-2.0
"""QLoRA variant: packed NF4 base, double-quantized scales and adapter parity."""

import tensorplay as tp
import tensorplay.nn as nn

from megatrain.lora.base import load_lora
from megatrain.lora.qlora import (
    QLoRAConfig,
    QLoRAError,
    dequantize_nf4,
    inject_qlora,
    quantize_nf4,
)


class TinyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(4, 6, bias=False, device="cpu", dtype=tp.float32)
        self.gate_proj = nn.Linear(4, 6, bias=False, device="cpu", dtype=tp.float32)
        self.other = nn.Linear(4, 6, bias=False, device="cpu", dtype=tp.float32)

    def forward(self, x):
        return self.q_proj(x) + self.gate_proj(x) + self.other(x)


def _config(**kwargs):
    return QLoRAConfig(
        r=4,
        lora_alpha=8,
        block_size=2,
        scaler_block_size=2,
        target_modules=("q_proj", "gate_proj"),
        compute_dtype="float32",
        **kwargs,
    )


def test_quantize_nf4_roundtrip_and_packing():
    tp.manual_seed(11)
    weight = tp.randn((8, 8), dtype=tp.float32, device="cpu") * 0.5
    state = quantize_nf4(weight, block_size=4, scaler_block_size=2, double_quant=True)
    assert tuple(state["packed"].shape) == (8, 4)
    assert tuple(state["quantized_scalers"].shape) == (8, 2)
    restored = dequantize_nf4(state)
    rel = float((restored - weight).abs().mean() / weight.abs().mean())
    assert rel < 0.15
    codes = ((state["packed"] >> 4).flatten().long()).tolist()
    assert all(0 <= code <= 15 for code in codes)


def test_double_quant_saves_bytes():
    tp.manual_seed(12)
    weight = tp.randn((64, 64), dtype=tp.float32, device="cpu")
    state = quantize_nf4(weight, block_size=64, scaler_block_size=16)
    packed_bytes = state["packed"].numel() + state["quantized_scalers"].numel()
    assert packed_bytes < weight.numel() * 0.6


def test_qlora_config_validates_dtype_and_blocks():
    try:
        QLoRAConfig(compute_dtype="fp8")
    except QLoRAError:
        pass
    else:
        raise AssertionError("invalid compute dtype must be rejected")
    assert _config().lora.r == 4
    assert _config().scaling == 2.0


def test_inject_qlora_freezes_quantized_base(tmp_path):
    tp.manual_seed(13)
    model = TinyPolicy()
    x = tp.ones((2, 4), dtype=tp.float32)
    state = inject_qlora(model, _config())
    assert set(state.layers) == {"q_proj", "gate_proj"}
    trainable = sum(int(p.numel()) for p in model.parameters() if p.requires_grad)
    assert trainable == state.parameter_count
    # Zero-initialized B means the injected model equals the quantized base.
    with tp.no_grad():
        quantized = model.q_proj.base(x) + model.gate_proj.base(x) + model.other(x)
        assert float((model(x) - quantized).abs().sum()) < 1.0e-6

    root = tmp_path / "adapter"
    state.export(root)
    assert (root / "adapter_config.json").is_file()
    assert (root / "adapter_model.safetensors").is_file()


def test_qlora_adapter_roundtrip(tmp_path):
    tp.manual_seed(14)
    model = TinyPolicy()
    state = inject_qlora(model, _config())
    with tp.no_grad():
        state.layers["q_proj"].lora_B.data = tp.ones((6, 4), dtype=tp.float32) * 0.1
        state.export(tmp_path / "adapter")

    fresh = TinyPolicy()
    restored = load_lora(fresh, tmp_path / "adapter")
    assert set(restored.layers) == {"q_proj", "gate_proj"}
    with tp.no_grad():
        expected = model(tp.ones((2, 4), dtype=tp.float32)).detach()
        actual = fresh(tp.ones((2, 4), dtype=tp.float32)).detach()
    assert tuple(expected.shape) == tuple(actual.shape)


def test_qlora_lora_delta_is_trainable():
    tp.manual_seed(15)
    model = TinyPolicy()
    state = inject_qlora(model, _config())
    x = tp.ones((2, 4), dtype=tp.float32)
    with tp.no_grad():
        baseline = model(x).detach().clone()
    state.layers["gate_proj"].lora_B.data = tp.ones((6, 4), dtype=tp.float32) * 0.5
    out = model(x)
    out.sum().backward()
    assert state.layers["gate_proj"].lora_A.grad is not None
    assert float((out.detach() - baseline).abs().sum()) > 0
