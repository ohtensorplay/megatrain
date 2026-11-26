# Copyright 2026 TensorPlay Authors
# SPDX-License-Identifier: Apache-2.0
"""LoRA training layers for TensorPlay (no PyTorch / PEFT dependency).

The injected layers keep the names of the original HuggingFace projections and
export an adapter that ``megalm.loading.lora.load_lora_adapter`` can serve, so
the same rank/alpha/targets work on both the training and rollout side.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tensorplay as tp
import tensorplay.nn as nn
import tensorplay.nn.functional as F
from tensorplay.nn import init as nn_init


class LoRAError(RuntimeError):
    """Raised for an invalid LoRA configuration or weight export."""


@dataclass(frozen=True, slots=True)
class LoraConfig:
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
    # rsLoRA (Kalajdzievski 2023), as exposed by PEFT / ROLL's
    # ``use_rslora``: scale by alpha / sqrt(r) instead of alpha / r.
    use_rslora: bool = False

    def __post_init__(self) -> None:
        if self.r < 1:
            raise LoRAError("lora rank must be positive")
        if self.lora_alpha < 1:
            raise LoRAError("lora_alpha must be positive")
        if not 0.0 <= self.lora_dropout < 1.0:
            raise LoRAError("lora_dropout must be in [0, 1)")
        if not self.target_modules:
            raise LoRAError("target_modules must not be empty")

    @property
    def scaling(self) -> float:
        if self.use_rslora:
            return self.lora_alpha / math.sqrt(self.r)
        return self.lora_alpha / self.r


class LoRALinear(nn.Module):
    """A frozen linear layer with trainable low-rank A/B factors."""

    def __init__(self, base: nn.Linear, config: LoraConfig) -> None:
        super().__init__()
        self.base = base
        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False
        in_features, out_features = int(base.weight.shape[1]), int(base.weight.shape[0])
        weight_dtype = base.weight.dtype
        self.lora_A = nn.Parameter(
            tp.zeros((config.r, in_features), dtype=weight_dtype, device=base.weight.device),
            requires_grad=True,
        )
        self.lora_B = nn.Parameter(
            tp.zeros((out_features, config.r), dtype=weight_dtype, device=base.weight.device),
            requires_grad=True,
        )
        # PEFT/loralib init: kaiming_uniform(a=sqrt(5)) for A, zeros for B.
        # Follows the upstream mcore_adapter LoRA layer initialization.
        # B stays zero so the injected model equals the base at step 0.
        nn_init.kaiming_uniform_(self.lora_A.data, a=5**0.5)
        nn_init.zeros_(self.lora_B.data)
        self.dropout = nn.Dropout(config.lora_dropout) if config.lora_dropout > 0 else None
        self.scaling = config.scaling

    def forward(self, x: Any) -> Any:
        result = self.base(x)
        source = self.dropout(x) if self.dropout is not None else x
        source = source.to(self.lora_A.dtype)
        delta = F.linear(F.linear(source, self.lora_A), self.lora_B)
        return result + delta * self.scaling


@dataclass(slots=True)
class LoRAState:
    config: LoraConfig
    layers: dict[str, LoRALinear] = field(default_factory=dict)

    @property
    def parameter_count(self) -> int:
        return sum(
            int(layer.lora_A.numel()) + int(layer.lora_B.numel()) for layer in self.layers.values()
        )

    def state_dict(self) -> dict[str, Any]:
        """PEFT-style names: ``<path>.lora_A.default.weight``."""
        out: dict[str, Any] = {}
        for name, layer in self.layers.items():
            out[f"{name}.lora_A.default.weight"] = layer.lora_A.data
            out[f"{name}.lora_B.default.weight"] = layer.lora_B.data
        return out

    def export(self, directory: str | Path) -> None:
        """Write an adapter readable by ``megalm.loading.lora.load_lora_adapter``."""
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
        }
        (root / "adapter_config.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        tp.save(self.state_dict(), root / "adapter_model.safetensors")


def _is_linear_like(module: Any) -> bool:
    """Accept ``nn.Linear`` and frozen quantized linears with a weight view."""
    if isinstance(module, nn.Linear):
        return True
    return (
        hasattr(module, "in_features")
        and hasattr(module, "out_features")
        and hasattr(module, "weight")
        and callable(module)
    )


def inject_lora(model: nn.Module, config: LoraConfig) -> LoRAState:
    """Replace ``config.target_modules`` linears in-place and freeze the base."""
    targets = set(config.target_modules)
    replaced: dict[str, LoRALinear] = {}
    for name, module in list(model.named_modules()):
        if name.rsplit(".", 1)[-1] not in targets:
            continue
        if not _is_linear_like(module) or isinstance(module, LoRALinear):
            continue
        parent = model
        parts = name.split(".")
        for part in parts[:-1]:
            parent = getattr(parent, part)
        wrapped = LoRALinear(module, config)
        setattr(parent, parts[-1], wrapped)
        replaced[name] = wrapped
    if not replaced:
        raise LoRAError(f"no target modules found for {sorted(targets)}")
    for parameter in model.parameters():
        parameter.requires_grad = False
    for layer in replaced.values():
        layer.lora_A.requires_grad = True
        layer.lora_B.requires_grad = True
    return LoRAState(config=config, layers=replaced)


def load_lora(model: nn.Module, directory: str | Path) -> LoRAState:
    """Load an exported adapter back into a model with injected LoRA layers."""
    root = Path(directory)
    manifest = json.loads((root / "adapter_config.json").read_text(encoding="utf-8"))
    config = LoraConfig(
        r=int(manifest["r"]),
        lora_alpha=int(manifest["lora_alpha"]),
        lora_dropout=float(manifest.get("lora_dropout", 0.0)),
        target_modules=tuple(manifest.get("target_modules", ())),
        use_rslora=bool(manifest.get("use_rslora", False)),
    )
    state = inject_lora(model, config)
    weights = tp.load(root / "adapter_model.safetensors", map_location="cpu")
    _assign(state, weights, strict=True)
    return state


def _assign(state: LoRAState, weights: Mapping[str, Any], *, strict: bool) -> None:
    missing: list[str] = []
    for name, layer in state.layers.items():
        for factor in ("A", "B"):
            key = f"{name}.lora_{factor}.default.weight"
            value = weights.get(key)
            if value is None:
                missing.append(key)
                continue
            target = layer.lora_A if factor == "A" else layer.lora_B
            target.data = value.to(device=target.device, dtype=target.dtype)
    if strict and missing:
        raise LoRAError(f"adapter is missing {len(missing)} tensors: {missing[:4]}")


__all__ = [
    "LoRAError",
    "LoRALinear",
    "LoRAState",
    "LoraConfig",
    "inject_lora",
    "load_lora",
]
