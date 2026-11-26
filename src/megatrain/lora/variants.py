# Copyright 2026 TensorPlay Authors
# SPDX-License-Identifier: Apache-2.0
"""LoRA optimizer variants: rsLoRA scaling helpers and LoRA+ dual learning rates."""

from __future__ import annotations

import math
from typing import Any

import tensorplay as tp

from .base import LoraConfig, LoRAError, LoRAState


def rslora_scaling(config: LoraConfig) -> float:
    """``alpha / sqrt(r)`` scaling used by rsLoRA (PEFT / ROLL ``use_rslora``)."""
    return config.lora_alpha / math.sqrt(config.r)


def lora_plus_optimizer(
    state: LoRAState,
    *,
    lr: float,
    ratio: float,
    weight_decay: float = 0.0,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    fused: bool = False,
) -> Any:
    """AdamW with LoRA+'s two rates: A at ``lr``, B at ``ratio * lr``.

    Mirrors Megatron-Bridge's ``get_lora_plus_config_overrides``: only the
    ``linear_out`` (B) factors get the multiplier; the base stays frozen.
    """
    if ratio <= 0:
        raise LoRAError("lora_plus_ratio must be positive")
    a_params = [layer.lora_A for layer in state.layers.values()]
    b_params = [layer.lora_B for layer in state.layers.values()]
    return tp.optim.AdamW(
        [
            {"params": a_params, "lr": lr},
            {"params": b_params, "lr": lr * ratio},
        ],
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
        fused=fused,
    )


__all__ = ["lora_plus_optimizer", "rslora_scaling"]
