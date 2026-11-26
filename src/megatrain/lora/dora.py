# Copyright 2026 TensorPlay Authors
# SPDX-License-Identifier: Apache-2.0
"""DoRA (Weight-Decomposed Low-Rank Adaptation) training variant.

Ported from the local Megatron-Bridge reference
(``peft/dora_layers.py::DoRALinear``): the pretrained weight plus its low-rank
update is decomposed into a trainable per-output-row magnitude and a normalized
direction.  The magnitude is initialized from the base weight norm and the
direction stays frozen at step zero, so injection is lossless.

The adapter export keeps the standard LoRA A/B tensors (readable by megaLM)
and adds the magnitude vectors with PEFT-style names.  megaLM's generic LoRA
loader does not yet apply the magnitude, so DoRA adapters are train/export
ready but not yet servable without a matching magnitude-aware kernel.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tensorplay as tp
import tensorplay.nn as nn
import tensorplay.nn.functional as F
from tensorplay.nn import init as nn_init

from .base import LoRAError


@dataclass(frozen=True, slots=True)
class DoRAConfig:
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    use_rslora: bool = False

    def __post_init__(self) -> None:
        if self.r < 1:
            raise LoRAError("DoRA rank must be positive")
        if self.lora_alpha < 1:
            raise LoRAError("lora_alpha must be positive")
        if not 0.0 <= self.lora_dropout < 1.0:
            raise LoRAError("lora_dropout must be in [0, 1)")
        if not self.target_modules:
            raise LoRAError("target_modules must not be empty")
        object.__setattr__(self, "target_modules", tuple(self.target_modules))

    @property
    def scaling(self) -> float:
        if self.use_rslora:
            return self.lora_alpha / math.sqrt(self.r)
        return self.lora_alpha / self.r


class DoRALinear(nn.Module):
    """One DoRA-wrapped linear layer (magnitude + normalized direction)."""

    def __init__(self, base: nn.Linear, config: DoRAConfig) -> None:
        super().__init__()
        self.base = base
        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False
        in_features = int(base.weight.shape[1])
        out_features = int(base.weight.shape[0])
        dtype = base.weight.dtype
        self.lora_A = nn.Parameter(
            tp.zeros((config.r, in_features), dtype=dtype, device=base.weight.device),
            requires_grad=True,
        )
        self.lora_B = nn.Parameter(
            tp.zeros((out_features, config.r), dtype=dtype, device=base.weight.device),
            requires_grad=True,
        )
        nn_init.kaiming_uniform_(self.lora_A.data, a=5**0.5)
        nn_init.zeros_(self.lora_B.data)
        weight_norm = (base.weight.detach().to(tp.float32) ** 2).sum(dim=1).sqrt().to(dtype)
        self.magnitude = nn.Parameter(weight_norm, requires_grad=True)
        self.dropout = nn.Dropout(config.lora_dropout) if config.lora_dropout > 0 else None
        self.scaling = config.scaling

    def _direction_norm(self) -> Any:
        weight = self.base.weight.detach().to(tp.float32)
        delta = (self.lora_B.to(tp.float32) @ self.lora_A.to(tp.float32)) * self.scaling
        return ((weight + delta) ** 2).sum(dim=1).sqrt().detach().to(self.magnitude.dtype)

    def forward(self, x: Any) -> Any:
        base_out = self.base(x)
        source = self.dropout(x) if self.dropout is not None else x
        source = source.to(self.lora_A.dtype)
        delta = F.linear(F.linear(source, self.lora_A), self.lora_B) * self.scaling
        scale = (self.magnitude / self._direction_norm()).reshape(1, 1, -1)
        result = scale * (base_out + delta)
        if self.training and self.dropout is not None:
            correction = self.base(source - x)
            result = result + (scale - 1.0) * correction
        return result


class DoRAState:
    """Injected DoRA layers plus a PEFT-shaped export."""

    def __init__(self, config: DoRAConfig, layers: dict[str, DoRALinear]) -> None:
        self.config = config
        self.layers = layers

    @property
    def parameter_count(self) -> int:
        return sum(
            int(p.numel())
            for layer in self.layers.values()
            for p in layer.parameters()
            if p.requires_grad
        )

    def state_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name, layer in self.layers.items():
            out[f"{name}.lora_A.default.weight"] = layer.lora_A.data
            out[f"{name}.lora_B.default.weight"] = layer.lora_B.data
            out[f"{name}.lora_magnitude_vector.default.weight"] = layer.magnitude.data
        return out

    def export(self, directory: str | Path) -> None:
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        config = self.config
        manifest = {
            "peft_type": "LORA",
            "r": config.r,
            "lora_alpha": config.lora_alpha,
            "lora_dropout": config.lora_dropout,
            "bias": "none",
            "task_type": "CAUSAL_LM",
            "target_modules": list(config.target_modules),
            "use_rslora": config.use_rslora,
            "use_dora": True,
        }
        (root / "adapter_config.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        tp.save(self.state_dict(), root / "adapter_model.safetensors")


def inject_dora(model: nn.Module, config: DoRAConfig | None = None) -> DoRAState:
    config = config or DoRAConfig()
    targets = set(config.target_modules)
    replaced: dict[str, DoRALinear] = {}
    for name, module in list(model.named_modules()):
        if name.rsplit(".", 1)[-1] not in targets:
            continue
        if not isinstance(module, nn.Linear) or isinstance(module, DoRALinear):
            continue
        parent = model
        parts = name.split(".")
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], DoRALinear(module, config))
        replaced[name] = getattr(parent, parts[-1])
    if not replaced:
        raise LoRAError(f"no DoRA target modules found for {sorted(targets)}")
    for parameter in model.parameters():
        parameter.requires_grad = False
    for layer in replaced.values():
        layer.lora_A.requires_grad = True
        layer.lora_B.requires_grad = True
        layer.magnitude.requires_grad = True
    return DoRAState(config=config, layers=replaced)


__all__ = ["DoRAConfig", "DoRALinear", "DoRAState", "inject_dora"]
