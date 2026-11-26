# Copyright 2026 TensorPlay Authors
# SPDX-License-Identifier: Apache-2.0
"""megaTrain: a generic TensorPlay training core.

The package owns *how* a training step runs (optimizer, FSDP, compile, gradient
accumulation, checkpoint/resume) and a small set of generic objectives. It does
not know about rollouts, rewards, policy versions or RL algorithms: those live
in the RL layer that depends on this package (e.g. megaRL).
"""

from .batch import TrainBatch, TrainSample, collate, pack_samples, slice_batch
from .config import TrainerConfig
from .kl import KL_ESTIMATORS, approx_kl
from .losses import (
    LOSS_TYPES,
    TensorBatch,
    sft_loss,
    sft_loss_with_head,
    token_logprobs,
)
from .stateless_adam import StatelessAdam
from .trainer import BaseTrainer

__all__ = [
    "BaseTrainer",
    "KL_ESTIMATORS",
    "LOSS_TYPES",
    "StatelessAdam",
    "TensorBatch",
    "TrainBatch",
    "TrainSample",
    "TrainerConfig",
    "approx_kl",
    "collate",
    "pack_samples",
    "sft_loss",
    "sft_loss_with_head",
    "slice_batch",
    "token_logprobs",
]
