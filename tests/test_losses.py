# Copyright 2026 TensorPlay Authors
# SPDX-License-Identifier: Apache-2.0
"""Objective helpers: token log-probs and distillation KL."""

import pytest
import tensorplay as tp
import tensorplay.nn.functional as F

from megatrain import BaseTrainer, TrainerConfig, TrainSample, approx_kl, collate, token_logprobs
from megatrain.losses import sft_loss
from megatrain.trainer import TensorBatch

from .conftest import TinyPolicy


def test_token_logprobs_are_normalized():
    logits = tp.tensor(
        [
            [[0.1, 2.0, -1.0], [1.0, 0.0, -2.0], [0.5, 0.5, 0.5]],
            [[1.0, 1.0, 1.0], [-1.0, 0.0, 2.0], [0.0, 3.0, -1.0]],
        ]
    )
    ids = tp.tensor([[0, 1, 2], [2, 0, 1]], dtype=tp.long)
    logps = token_logprobs(logits, ids)
    assert tuple(logps.shape) == (2, 3)
    expected = tp.gather(F.log_softmax(logits, dim=-1), -1, ids.unsqueeze(-1)).squeeze(-1)
    assert float((logps - expected).abs().sum().flatten()[0]) == pytest.approx(0.0, abs=1e-6)
    assert all(float(v) <= 1.0 + 1e-6 for v in tp.exp(logps).flatten().tolist())


@pytest.mark.parametrize("est,minimum", [("k2", 0.0), ("k3", 0.0), ("low_var_kl", 0.0)])
def test_kl_estimators_are_nonnegative(est, minimum):
    a = tp.tensor([0.2, -0.5, 1.0])
    b = tp.tensor([-0.1, 0.4, 0.9])
    value = approx_kl(a, b, kl_type=est)
    assert all(float(v) >= minimum - 1e-6 for v in value.flatten().tolist())


def test_distillation_updates_student_only():
    tp.manual_seed(7)
    student = TinyPolicy()
    teacher = TinyPolicy()
    trainer = BaseTrainer(
        student,
        TrainerConfig(learning_rate=1e-2),
        teacher_model=teacher,
    )
    batch = collate([TrainSample([1, 2, 3, 4], [0, 0, 1, 1])])
    tensors = TensorBatch(batch, device="cpu")
    with tp.no_grad():
        teacher_logps = trainer._teacher_logprobs(tensors)
    loss, diag = sft_loss(
        trainer._forward(tensors),
        tensors,
        teacher_logprobs=teacher_logps,
        distill_coef=0.5,
        kl_type="k3",
    )
    assert "kl" in diag
    loss.backward()
    assert student.head.weight.grad is not None
    assert float(student.head.weight.grad.abs().sum()) > 0
    assert teacher.head.weight.grad is None


def test_teacher_logprobs_are_computed_once_for_all_microbatches():
    tp.manual_seed(7)
    student = TinyPolicy()
    teacher = TinyPolicy()
    trainer = BaseTrainer(
        student,
        TrainerConfig(learning_rate=1e-2, gradient_accumulation_steps=2),
        teacher_model=teacher,
    )
    batch = collate([TrainSample([1, 2, 3, 4], [0, 0, 1, 1]) for _ in range(2)])
    metrics = trainer.train_step(batch, loss_kwargs={"distill_coef": 0.5})
    assert "train/kl" in metrics
    assert teacher.head.weight.grad is None
