# Copyright 2026 TensorPlay Authors
# SPDX-License-Identifier: Apache-2.0
"""LoRA variants for TensorPlay training (no PyTorch / PEFT dependency).

Layout:
    base     -- canonical LoRA: config, injection, freeze, adapter export/load
    qlora    -- frozen NF4 base + bf16 LoRA (QLoRA variant)
    dora     -- magnitude/direction decomposition (DoRA variant)
    variants -- rsLoRA scaling helper and LoRA+ dual learning rates
"""

from .base import LoraConfig, LoRAError, LoRALinear, LoRAState, inject_lora, load_lora
from .dora import DoRAConfig, DoRALinear, DoRAState, inject_dora
from .qlora import (
    NF4Linear,
    QLoRAConfig,
    QLoRAError,
    dequantize_nf4,
    inject_qlora,
    quantize_nf4,
)
from .variants import lora_plus_optimizer, rslora_scaling

__all__ = [
    "DoRAConfig",
    "DoRALinear",
    "DoRAState",
    "LoRAError",
    "LoRALinear",
    "LoRAState",
    "LoraConfig",
    "NF4Linear",
    "QLoRAConfig",
    "QLoRAError",
    "dequantize_nf4",
    "inject_dora",
    "inject_lora",
    "inject_qlora",
    "load_lora",
    "lora_plus_optimizer",
    "quantize_nf4",
    "rslora_scaling",
]
