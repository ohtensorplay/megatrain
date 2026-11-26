# Copyright 2026 TensorPlay Authors
# SPDX-License-Identifier: Apache-2.0
"""QLoRA training variant: a frozen NF4 base with bf16 LoRA adapters.

This is a LoRA variant, not a standalone quantizer: it follows the QLoRA
recipe (4-bit NormalFloat weights, double-quantized block scales, bf16 compute
and trainable high-precision LoRA factors) but exposes exactly the same
adapter contract as :mod:`megatrain.lora`, so the exported adapter loads into
``megalm.loading.lora`` unchanged.

The NF4 codebook and double-quantization scheme mirror the vendored reference
in ``tensorplay-main/third_party/torchao/.../nf4/nf4_tensor.py``: codes are
packed two per byte (high nibble first), scales are quantized to int8 with one
floating-point factor per scaler block.  Everything is implemented with
TensorPlay operations only; no PyTorch / PEFT / bitsandbytes is involved.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import tensorplay as tp
import tensorplay.nn as nn

from .base import LoraConfig, LoRAError, LoRAState, inject_lora

# Table 1 / Appendix A of the QLoRA paper, as used by the vendored torchao
# reference.  Unlike the commonly quoted [-1, 0.696, ...] list this one spans
# [-1, 1] so block absmax scaling uses the full code range.
_NF4_CODEBOOK = (
    -1.0000,
    -0.6962,
    -0.5251,
    -0.3949,
    -0.2844,
    -0.1848,
    -0.0911,
    0.0000,
    0.0796,
    0.1609,
    0.2461,
    0.3379,
    0.4407,
    0.5626,
    0.7230,
    1.0000,
)


class QLoRAError(LoRAError):
    """Raised for an invalid QLoRA configuration or quantized weight."""


@dataclass(frozen=True, slots=True)
class QLoRAConfig:
    """LoRA configuration plus the QLoRA base-quantization knobs."""

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
    block_size: int = 64
    scaler_block_size: int = 16
    double_quant: bool = True
    compute_dtype: str = "bfloat16"
    use_rslora: bool = False

    def __post_init__(self) -> None:
        if self.block_size <= 0 or self.block_size % 2 != 0:
            raise QLoRAError("block_size must be a positive even integer")
        if self.scaler_block_size <= 0:
            raise QLoRAError("scaler_block_size must be positive")
        if self.compute_dtype not in {"float16", "bfloat16", "float32"}:
            raise QLoRAError(f"unsupported compute dtype: {self.compute_dtype}")
        object.__setattr__(self, "target_modules", tuple(self.target_modules))

    @property
    def lora(self) -> LoraConfig:
        return LoraConfig(
            r=self.r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=self.target_modules,
            use_rslora=self.use_rslora,
        )

    @property
    def scaling(self) -> float:
        return self.lora.scaling

    @staticmethod
    def codebook(dtype: Any, device: Any) -> Any:
        return tp.tensor(_NF4_CODEBOOK, dtype=dtype, device=device)


def quantize_nf4(
    weight: Any,
    *,
    block_size: int = 64,
    scaler_block_size: int = 16,
    double_quant: bool = True,
) -> dict[str, Any]:
    """Quantize a 2-D weight into QLoRA's packed NF4 representation."""
    if weight.ndim != 2:
        raise QLoRAError("NF4 quantization expects a 2-D weight")
    rows, columns = tuple(weight.shape)
    if columns % block_size != 0:
        raise QLoRAError(f"in_features {columns} is not divisible by block_size {block_size}")
    n_blocks = columns // block_size
    work_dtype = tp.float32
    w = weight.detach().to(work_dtype)
    blocks = w.reshape(rows, n_blocks, block_size)
    absmax = blocks.abs().amax(dim=2)
    device = w.device

    if double_quant:
        scaler_block_size = min(scaler_block_size, n_blocks)
        if n_blocks % scaler_block_size != 0:
            raise QLoRAError(
                f"scale blocks {n_blocks} is not divisible by scaler_block_size {scaler_block_size}"
            )
        scaler_mean = absmax.mean()
        centered = absmax - scaler_mean
        n_scaler_blocks = n_blocks // scaler_block_size
        scaler_blocks = centered.reshape(rows, n_scaler_blocks, scaler_block_size)
        scaler_absmax = scaler_blocks.abs().amax(dim=2).clamp(min=1.0e-8)
        quantization_factor = 256.0 / (2.0 * scaler_absmax)
        quantized_scalers = (
            (scaler_blocks * quantization_factor.unsqueeze(-1))
            .round()
            .clamp(-128, 127)
            .to(tp.int8)
            .reshape(rows, n_blocks)
        )
    else:
        scaler_mean = tp.zeros((), dtype=work_dtype, device=device)
        n_scaler_blocks = n_blocks
        quantization_factor = tp.ones((rows, n_scaler_blocks), dtype=work_dtype, device=device)
        quantized_scalers = absmax.to(tp.int8)

    # Nearest NF4 code per element, then pack two consecutive codes per byte
    # with the first code in the high nibble (torchao reference layout).
    codebook = QLoRAConfig.codebook(work_dtype, device)
    scaled = blocks / absmax.unsqueeze(-1).clamp(min=1.0e-12)
    distances = (scaled.unsqueeze(-1) - codebook.reshape(1, 1, 1, 16)).abs()
    codes = distances.argmin(dim=-1).to(tp.uint8)
    packed = ((codes[:, :, 0::2] << 4) | codes[:, :, 1::2]).reshape(rows, columns // 2)

    return {
        "packed": packed.contiguous(),
        "quantized_scalers": quantized_scalers.contiguous(),
        "quantization_factor": quantization_factor.contiguous(),
        "scaler_mean": scaler_mean,
        "in_features": columns,
        "out_features": rows,
        "block_size": block_size,
        "scaler_block_size": scaler_block_size,
        "double_quant": double_quant,
    }


def dequantize_nf4(state: dict[str, Any], *, dtype: Any | None = None) -> Any:
    """Rebuild a dense weight from a packed NF4 state dict."""
    rows = state["out_features"]
    columns = state["in_features"]
    block = state["block_size"]
    n_blocks = columns // block
    work_dtype = tp.float32
    device = state["packed"].device

    if state["double_quant"]:
        sb = state["scaler_block_size"]
        n_scaler_blocks = n_blocks // sb
        scales = state["quantized_scalers"].to(work_dtype).reshape(rows, n_scaler_blocks, sb)
        scales = scales / state["quantization_factor"].unsqueeze(-1)
        absmax = (scales.reshape(rows, n_blocks) + state["scaler_mean"]).clamp(min=1.0e-8)
    else:
        absmax = state["quantized_scalers"].to(work_dtype).clamp(min=1.0e-8)

    packed = state["packed"].reshape(rows, n_blocks, block // 2)
    # Unpack the two nibbles as uint8 *before* widening to long: stacking the
    # small uint8 tensors is ~2x cheaper than stacking two fp32 gather results.
    codes = tp.stack(((packed >> 4), (packed & 15)), dim=-1).reshape(rows, columns).long()
    codebook = QLoRAConfig.codebook(work_dtype, device)
    values = tp.take(codebook, codes)
    dense = values * absmax.repeat_interleave(block, dim=1)
    return dense if dtype is None else dense.to(dtype)


class NF4Linear(nn.Module):
    """Frozen linear layer storing its weight in packed NF4 form."""

    def __init__(self, state: dict[str, Any], *, dtype: Any, bias: Any | None = None) -> None:
        super().__init__()
        self.in_features = int(state["in_features"])
        self.out_features = int(state["out_features"])
        self.block_size = int(state["block_size"])
        self.scaler_block_size = int(state["scaler_block_size"])
        self.double_quant = bool(state["double_quant"])
        self.compute_dtype = dtype
        self.register_buffer("nf4_packed", state["packed"])
        self.register_buffer("nf4_scalers", state["quantized_scalers"])
        self.register_buffer("nf4_factor", state["quantization_factor"])
        self.register_buffer("nf4_mean", state["scaler_mean"])
        self.register_buffer(
            "nf4_codebook", QLoRAConfig.codebook(tp.float32, state["packed"].device)
        )
        if bias is not None:
            self.register_buffer("nf4_bias", bias.detach().to(dtype).contiguous())

    @classmethod
    def from_linear(cls, linear: nn.Linear, config: QLoRAConfig) -> NF4Linear:
        dtype = getattr(tp, config.compute_dtype)
        state = quantize_nf4(
            linear.weight,
            block_size=config.block_size,
            scaler_block_size=config.scaler_block_size,
            double_quant=config.double_quant,
        )
        return cls(state, dtype=dtype, bias=linear.bias)

    def state(self) -> dict[str, Any]:
        return {
            "packed": self.nf4_packed,
            "quantized_scalers": self.nf4_scalers,
            "quantization_factor": self.nf4_factor,
            "scaler_mean": self.nf4_mean,
            "in_features": self.in_features,
            "out_features": self.out_features,
            "block_size": self.block_size,
            "scaler_block_size": self.scaler_block_size,
            "double_quant": self.double_quant,
        }

    @property
    def weight(self) -> Any:
        """Dense compute view; used by ``LoRALinear`` for shape/dtype only."""
        return dequantize_nf4(self.state(), dtype=self.compute_dtype)

    @property
    def bias(self) -> Any | None:
        """``LoRALinear`` inspects this attribute to decide about bias freezing."""
        return getattr(self, "nf4_bias", None)

    def forward(self, x: Any) -> Any:
        # Fused NF4 dequant+matmul: the packed weight stays resident and no
        # dense copy is saved for backward, so gradients reach every LoRA
        # target (unlike a no_grad base, which silently kills q/k/v/gate/up).
        from megalm.kernels.nf4_gemm import nf4_linear_scratch

        leading = tuple(x.shape[:-1])
        flat = x.reshape(-1, x.shape[-1]).contiguous()
        out = nf4_linear_scratch(
            flat,
            self.nf4_packed,
            self.nf4_scalers,
            self.nf4_factor,
            self.nf4_mean,
            self.nf4_codebook,
            block=self.block_size,
            sb=self.scaler_block_size,
            double_quant=self.double_quant,
        ).reshape(*leading, self.out_features)
        if hasattr(self, "nf4_bias"):
            out = out + self.nf4_bias
        return out


def inject_qlora(model: nn.Module, config: QLoRAConfig | None = None) -> LoRAState:
    """Quantize the target linears to NF4, then inject trainable LoRA factors."""
    config = config or QLoRAConfig()
    targets = set(config.target_modules)
    replaced = 0
    for name, module in list(model.named_modules()):
        if name.rsplit(".", 1)[-1] not in targets:
            continue
        if not isinstance(module, nn.Linear) or isinstance(module, NF4Linear):
            continue
        parent = model
        parts = name.split(".")
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], NF4Linear.from_linear(module, config))
        replaced += 1
    if replaced == 0:
        raise QLoRAError(f"no target modules found for {sorted(targets)}")
    return inject_lora(model, config.lora)


__all__ = [
    "NF4Linear",
    "QLoRAConfig",
    "QLoRAError",
    "dequantize_nf4",
    "inject_qlora",
    "quantize_nf4",
]
