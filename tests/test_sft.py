"""Pretraining / SFT step: real forward, backward and optimizer step."""

import math

import pytest
import tensorplay as tp
import tensorplay.nn.functional as F

from megatrain import BaseTrainer, TrainerConfig, TrainSample, collate
from megatrain.losses import sft_loss
from megatrain.trainer import TensorBatch

from .conftest import TinyPolicy


def make_batch():
    # Completion starts at index 2 / 1 respectively; prompt and pad are masked.
    return collate(
        [
            TrainSample([1, 2, 3, 4], [0, 0, 1, 1]),
            TrainSample([5, 6, 7], [0, 1, 1]),
        ]
    )


def make_trainer(policy, **kwargs):
    return BaseTrainer(policy, TrainerConfig(learning_rate=1e-2, **kwargs))


def test_sft_step_produces_gradients_and_updates_weights(policy):
    trainer = make_trainer(policy)
    before = trainer.policy.head.weight.detach().clone()
    metrics = trainer.train_step(make_batch(), dry_run=True)
    assert trainer.policy.head.weight.grad is not None
    assert any(abs(float(v)) > 0 for v in trainer.policy.head.weight.grad.flatten().tolist())
    assert math.isfinite(metrics["train/loss"])
    trainer.train_step(make_batch())
    after = trainer.policy.head.weight.detach()
    assert float((after - before).abs().sum()) > 0
    assert trainer.step_count == 1
    assert metrics["train/step"] == 0.0


def test_gradient_accumulation_matches_single_batch():
    def grads(accum: int):
        tp.manual_seed(1234)
        trainer = make_trainer(TinyPolicy(), gradient_accumulation_steps=accum)
        batch = collate([TrainSample([1, 2, 3, 4], [0, 0, 1, 1]) for _ in range(4)])
        trainer.train_step(batch, dry_run=True)
        return [float(v) for v in trainer.policy.head.weight.grad.flatten().tolist()]

    single, accumulated = grads(1), grads(2)
    assert single == pytest.approx(accumulated, abs=1e-5)


def test_sft_loss_matches_manual_masked_exact_reference(policy):
    batch = make_batch()
    trainer = make_trainer(policy)
    tensors = TensorBatch(batch, device="cpu")
    logits = trainer._forward(tensors)
    loss, _ = sft_loss(logits, tensors)
    expected_ce = F.cross_entropy(
        logits[:, :-1, :].reshape(-1, logits.shape[-1]),
        tensors.input_ids[:, 1:].reshape(-1),
        reduction="none",
    ).reshape(tensors.loss_mask[:, 1:].shape)
    keep = tensors.loss_mask[:, 1:]
    expected = (expected_ce * keep).sum() / keep.sum()
    assert float(loss.detach().flatten()[0]) == pytest.approx(
        float(expected.detach().flatten()[0]), abs=1e-6
    )


def test_masked_out_tokens_produce_zero_loss(policy):
    trainer = make_trainer(policy)
    batch = collate([TrainSample([1, 2, 3], [0, 0, 0])])
    metrics = trainer.train_step(batch, dry_run=True)
    assert metrics["train/loss"] == pytest.approx(0.0)
    grad = trainer.policy.head.weight.grad
    assert grad is None or float(grad.abs().sum()) == pytest.approx(0.0)
