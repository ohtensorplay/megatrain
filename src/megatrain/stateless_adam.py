# Copyright 2026 TensorPlay Authors
# SPDX-License-Identifier: Apache-2.0
"""Stateless Adam/AdamW for one-step optimizers.

Adapted from megaRL's ``stateless_adam.py`` (Apache-2.0). A stateless optimizer
resets its first and second moments at every step, so it never persists moment
tensors. The update is exactly Adam/AdamW with zero initial moments, which
removes optimizer moments from checkpoints at the cost of bit-exact resume.

Reference: ``upstream-checkouts/slime/slime/backends/megatron_utils/stateless_adam.py``.
"""

from __future__ import annotations

import math
from typing import Any

import tensorplay as tp


class StatelessAdam(tp.optim.Optimizer):
    """AdamW without persistent ``exp_avg`` / ``exp_avg_sq`` tensors."""

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        *,
        bias_correction: bool = True,
        adam_w_mode: bool = True,
        maximize: bool = False,
    ) -> None:
        if lr < 0:
            raise ValueError("learning rate must be non-negative")
        if eps < 0:
            raise ValueError("eps must be non-negative")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError("betas must be in [0, 1)")
        defaults = dict(
            lr=lr,
            betas=tuple(betas),
            eps=eps,
            weight_decay=weight_decay,
            bias_correction=bias_correction,
            adam_w_mode=adam_w_mode,
            maximize=maximize,
        )
        super().__init__(params, defaults)
        for group in self.param_groups:
            group.setdefault("step", 0)

    def state_dict(self) -> dict:
        """Only hyperparameters and scheduler progress; no moment tensors."""
        state = super().state_dict()
        state["state"] = {}
        state["stateless"] = True
        return state

    def load_state_dict(self, state_dict: dict) -> None:
        """Restore hyperparameters, ignoring any moment tensors in the input."""
        restored = dict(state_dict)
        restored["state"] = {}
        restored.pop("stateless", None)
        super().load_state_dict(restored)

    @tp.no_grad()
    def step(self, closure: Any | None = None):
        loss = None
        if closure is not None:
            with tp.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            adam_w = group["adam_w_mode"]
            bias_correction = group["bias_correction"]
            maximize = group["maximize"]
            if bias_correction:
                numerator_scale = 1.0
                denominator_scale = 1.0
            else:
                numerator_scale = 1.0 - beta1
                denominator_scale = math.sqrt(1.0 - beta2)
            group["step"] = 1
            for param in group["params"]:
                grad = param.grad
                if grad is None:
                    continue
                grad_for_update = grad.neg() if maximize else grad
                if weight_decay:
                    if adam_w:
                        param.mul_(1.0 - lr * weight_decay)
                    else:
                        grad_for_update = grad_for_update + param * weight_decay
                denom = grad_for_update.abs() * denominator_scale + eps
                param.addcdiv_(grad_for_update, denom, value=-lr * numerator_scale)
        return loss


__all__ = ["StatelessAdam"]
