# Copyright 2026 TensorPlay Authors
# SPDX-License-Identifier: Apache-2.0
"""Training-only MiniCPM4 layers built on the shared megaLM model definition.

megaLM owns the checkpoint contract: :class:`GenericCausalLMConfig` parses
``config.json`` (MiniCPM scales + LongRoPE) and :class:`GenericRotaryCache`
materializes the cos/sin tables. This module adds only the autograd-friendly
forward path that SFT needs; serving keeps using megaLM's fused paged-KV path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import tensorplay as tp
import tensorplay.nn as nn
import tensorplay.nn.functional as F
from megalm.models.generic_causal_lm.config import GenericCausalLMConfig as MiniCPM4Config
from megalm.models.generic_causal_lm.model import GenericRotaryCache as MiniCPMRotary


class MiniCPM4Error(RuntimeError):
    """Raised for an invalid MiniCPM4 checkpoint for training."""


class TrainingRotary(MiniCPMRotary):
    """Apply the shared megaLM cos/sin tables in an autograd-friendly way."""

    def apply(self, x: tp.Tensor, position_ids: tp.Tensor) -> tp.Tensor:
        positions = position_ids.reshape(-1)
        cos = self.cos.index_select(0, positions).to(x.dtype)
        sin = self.sin.index_select(0, positions).to(x.dtype)
        shape = (x.shape[0], 1, x.shape[-2], x.shape[-1] // 2)
        cos, sin = cos.reshape(shape), sin.reshape(shape)
        first, second = x[..., : shape[-1]], x[..., shape[-1] :]
        return tp.cat(
            (first * cos - second * sin, second * cos + first * sin),
            dim=-1,
        )


_CAUSAL_MASKS: dict[tuple[int, Any, Any], Any] = {}


def _causal_mask(length: int, dtype: Any, device: Any) -> Any:
    key = (length, str(dtype), str(device))
    mask = _CAUSAL_MASKS.get(key)
    if mask is None:
        rows = [[0.0 if j <= i else -1e4 for j in range(length)] for i in range(length)]
        mask = tp.tensor(rows, dtype=dtype).to(device)
        _CAUSAL_MASKS[key] = mask
    return mask


class MiniCPM4Attention(nn.Module):
    def __init__(self, config: MiniCPM4Config, *, device: Any, dtype: Any) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.q_proj = nn.Linear(
            config.hidden_size,
            self.num_heads * self.head_dim,
            bias=config.qkv_bias,
            device=device,
            dtype=dtype,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=config.qkv_bias,
            device=device,
            dtype=dtype,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=config.qkv_bias,
            device=device,
            dtype=dtype,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        hidden: tp.Tensor,
        rotary: TrainingRotary,
        position_ids: tp.Tensor,
        attention_mask: tp.Tensor,
    ) -> tp.Tensor:
        batch, length, _ = hidden.shape
        q = (
            self.q_proj(hidden)
            .reshape(batch, length, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(hidden)
            .reshape(batch, length, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(hidden)
            .reshape(batch, length, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        q, k = rotary.apply(q, position_ids), rotary.apply(k, position_ids)
        causal = _causal_mask(length, hidden.dtype, hidden.device)
        padding = (1.0 - attention_mask.to(hidden.dtype)) * -1e4
        mask = causal.reshape(1, 1, length, length) + padding.reshape(batch, 1, 1, length)
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True)
        return self.o_proj(attended.transpose(1, 2).reshape(batch, length, -1))


class MiniCPM4MLP(nn.Module):
    def __init__(self, config: MiniCPM4Config, *, device: Any, dtype: Any) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False, device=device, dtype=dtype
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False, device=device, dtype=dtype
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False, device=device, dtype=dtype
        )

    def forward(self, x: tp.Tensor) -> tp.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MiniCPM4DecoderLayer(nn.Module):
    def __init__(self, config: MiniCPM4Config, *, device: Any, dtype: Any) -> None:
        super().__init__()
        self.input_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, device=device, dtype=dtype
        )
        self.self_attn = MiniCPM4Attention(config, device=device, dtype=dtype)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, device=device, dtype=dtype
        )
        self.mlp = MiniCPM4MLP(config, device=device, dtype=dtype)
        self.residual_multiplier = config.residual_multiplier

    def forward(
        self,
        hidden: tp.Tensor,
        rotary: TrainingRotary,
        position_ids: tp.Tensor,
        attention_mask: tp.Tensor,
    ) -> tp.Tensor:
        residual = hidden
        hidden = self.input_layernorm(hidden)
        hidden = self.self_attn(hidden, rotary, position_ids, attention_mask)
        hidden = residual + hidden * self.residual_multiplier
        residual = hidden
        hidden = self.post_attention_layernorm(hidden)
        hidden = self.mlp(hidden)
        return residual + hidden * self.residual_multiplier


class MiniCPM4ForCausalLM(nn.Module):
    """Training model returning ``[B, L, vocab]`` logits for ``sft_loss``."""

    def __init__(
        self,
        config: MiniCPM4Config,
        *,
        device: Any = "cuda",
        dtype: Any = tp.bfloat16,
    ) -> None:
        super().__init__()
        self.config = config
        self.device = tp.device(device)
        self.dtype = dtype
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, device=device, dtype=dtype
        )
        self.rotary = TrainingRotary(config, device=device)
        self.layers = nn.ModuleList(
            [
                MiniCPM4DecoderLayer(config, device=device, dtype=dtype)
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.norm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, device=device, dtype=dtype
        )

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Path,
        *,
        device: Any = "cuda",
        dtype: Any = tp.bfloat16,
    ) -> tuple[MiniCPM4ForCausalLM, dict[str, int]]:
        config = MiniCPM4Config.from_path(model_path)
        if config.model_type != "minicpm":
            raise MiniCPM4Error("checkpoint is not a MiniCPMForCausalLM model")
        model = cls(config, device=device, dtype=dtype)
        return model, load_weights(model, model_path)

    def forward(self, batch: Any) -> tp.Tensor:
        input_ids = batch.input_ids
        attention_mask = batch.attention_mask
        position_ids = tp.broadcast_to(
            tp.arange(0, input_ids.shape[1], dtype=tp.long).to(input_ids.device).reshape(1, -1),
            input_ids.shape,
        )
        hidden = self.embed_tokens(input_ids) * self.config.embedding_multiplier
        for layer in self.layers:
            hidden = layer(hidden, self.rotary, position_ids, attention_mask)
        hidden = self.norm(hidden)
        # MiniCPM4 ties lm_head to the embedding table.
        return F.linear(hidden, self.embed_tokens.weight) * self.config.logit_scale


def load_weights(model: MiniCPM4ForCausalLM, model_path: str | Path) -> dict[str, int]:
    """Load the HF safetensors checkpoint via TensorPlay serialization."""
    root = Path(model_path)
    shards = sorted(root.glob("*.safetensors"))
    if not shards:
        raise MiniCPM4Error(f"no safetensors shards in {root}")
    params = dict(model.named_parameters())
    loaded = 0
    unexpected: list[str] = []
    for shard in shards:
        weights = tp.load(str(shard), map_location="cpu", mmap=True)
        for name, value in weights.items():
            target = name[len("model.") :] if name.startswith("model.") else name
            parameter = params.get(target)
            if parameter is None:
                if name != "lm_head.weight":
                    unexpected.append(name)
                continue
            if tuple(parameter.shape) != tuple(value.shape):
                raise MiniCPM4Error(
                    f"shape mismatch for {name}: {tuple(parameter.shape)} vs {tuple(value.shape)}"
                )
            parameter.data = value.to(device=model.device, dtype=model.dtype)
            loaded += 1
    if loaded != len(params):
        raise MiniCPM4Error(f"loaded {loaded}/{len(params)} parameters")
    return {"loaded": loaded, "shards": len(shards), "unexpected": len(unexpected)}


__all__ = ["MiniCPM4Config", "MiniCPM4ForCausalLM", "TrainingRotary", "load_weights"]
