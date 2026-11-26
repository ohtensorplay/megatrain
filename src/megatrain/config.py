"""Backend-only training configuration.

Nothing in here describes an RL objective: loss selection is a per-step
argument to :meth:`megatrain.BaseTrainer.train_step`, so the same trainer
serves pretraining, SFT, distillation and RL post-training.
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True, slots=True)
class TrainerConfig:
    device: str = "cpu"
    gradient_accumulation_steps: int = 1
    """Number of microbatches per optimizer step (no_sync on all but the last)."""

    optimizer_kind: str = "adamw"
    """adamw | stateless_adam."""
    learning_rate: float = 1e-5
    adam_betas: tuple[float, float] = (0.9, 0.999)
    adam_eps: float = 1e-8
    weight_decay: float = 0.0
    stateless_bias_correction: bool = True
    max_grad_norm: float | None = None
    """Global L2 clip before optimizer.step(); None disables clipping."""

    fused_optimizer: bool = False
    """Use TP's fused multi-tensor AdamW (CUDA only).

    ``fused=True, capturable=True`` was 2.83 -> 2.74 ms per step at 160M
    parameters, and it keeps the step capturable by CUDA graphs. Ignored on CPU.
    """

    amp: bool = False
    """Autocast + GradScaler. Automatically inactive on CPU."""
    amp_dtype: str = "float16"
    """float16 (with GradScaler) | bfloat16 (no scaler, needs sm80+).
    bfloat16 avoids scaling/unscaling and is usually faster on Ampere+."""
    ce_chunk_tokens: int | None = None
    """Cap cross-entropy rows per chunk to bound peak log_softmax memory."""

    compile_policy: bool = False
    # tp.compile(dynamic=True) falls back to the graph executor for this model and
    # measured slower + 3.7 GiB heavier than dynamic=False on fixed-shape GPT
    # (84.5k vs 74.9k tok/s). Turn it on only for 真正的 varlen/packed batches.
    compile_dynamic: bool = False
    fsdp: bool = False
    seed: int | None = None
    """When set, seed Python/TensorPlay/numpy RNGs in __init__."""

    checkpoint_dir: str | None = None
    max_checkpoints: int = 3


__all__ = ["TrainerConfig"]
