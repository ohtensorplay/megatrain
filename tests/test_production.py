"""Production-path checks: packing, clipping, scheduler, resumable fit loop."""

import pytest
import tensorplay as tp

from megatrain import (
    BaseTrainer,
    TrainerConfig,
    TrainSample,
    collate,
    pack_samples,
)

from .conftest import TinyPolicy


class TinyScheduler:
    def __init__(self) -> None:
        self.step_count = 0

    def step(self) -> None:
        self.step_count += 1

    def state_dict(self) -> dict:
        return {"step_count": self.step_count}

    def load_state_dict(self, state: dict) -> None:
        self.step_count = int(state["step_count"])


def samples():
    # Token ids stay inside TinyPolicy's 8-token vocabulary.
    return [
        TrainSample([i % 8 for i in range(10)], [0] * 3 + [1] * 7),
        TrainSample([i % 8 for i in range(20, 26)], [0, 1, 1, 1, 1, 1]),
        TrainSample([i % 8 for i in range(30, 34)], [0, 0, 1, 1]),
    ]


def test_pack_samples_respects_token_budget_and_alignment():
    batches = pack_samples(samples(), max_tokens=8, alignment=2, shuffle=False)
    assert batches
    assert all(batch.token_count <= 8 for batch in batches)
    assert all(batch.cu_seqlens[-1] % 2 == 0 for batch in batches)
    # 10 -> 8+2, 6 -> 6, 4 -> 4: nothing is silently dropped.
    assert sum(batch.token_count for batch in batches) == 20


def test_pack_samples_deterministic_with_seed():
    first = [b.input_ids for b in pack_samples(samples(), 8, seed=7)]
    second = [b.input_ids for b in pack_samples(samples(), 8, seed=7)]
    assert first == second


def test_gradient_clipping_is_enforced_and_reported(policy):
    trainer = BaseTrainer(policy, TrainerConfig(learning_rate=1e-2, max_grad_norm=1e-3))
    batch = pack_samples(samples(), 8, shuffle=False)[0]
    metrics = trainer.train_step(batch, dry_run=True)
    assert "train/grad_norm" in metrics  # pre-clip norm, reported for telemetry
    total_sq = sum(
        float((p.grad.detach() ** 2).sum().flatten()[0])
        for p in trainer.policy.parameters()
        if p.grad is not None
    )
    assert total_sq ** 0.5 <= 1e-3 + 1e-5


def test_scheduler_is_stepped_and_checkpointed(tmp_path):
    tp_scheduler = TinyScheduler()
    trainer = BaseTrainer(
        TinyPolicy(), TrainerConfig(learning_rate=1e-2), scheduler=tp_scheduler
    )
    batch = pack_samples(samples(), 8, shuffle=False)[0]
    trainer.train_step(batch)
    trainer.train_step(batch)
    assert tp_scheduler.step_count == 2
    path = trainer.save_checkpoint(tmp_path / "v1")
    restored = TinyScheduler()
    fresh = BaseTrainer(TinyPolicy(), scheduler=restored)
    fresh.load_checkpoint(path)
    assert restored.step_count == 2
    assert fresh.step_count == 2


def test_fit_streams_metrics_and_publishes_checkpoints(tmp_path):
    trainer = BaseTrainer(
        TinyPolicy(),
        TrainerConfig(learning_rate=1e-2, checkpoint_dir=str(tmp_path)),
    )
    metrics = list(
        trainer.fit(samples(), max_tokens=8, epochs=2, seed=3, checkpoint_every=1)
    )
    assert len(metrics) >= 2
    assert trainer.step_count == len(metrics)
    assert (tmp_path / f"step-{trainer.step_count}").exists()


class HookTrainer(BaseTrainer):
    """Stand-in for the RL subclass that overrides batch/normalization hooks."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sliced = 0

    def _slice_batch(self, batch, start, stop):
        self.sliced += 1
        return super()._slice_batch(batch, start, stop)

    def _keep_tokens(self, batch) -> int:
        return 1  # e.g. a sequence-normalized RL objective


def test_subclass_hooks_are_used_for_rl_integration(policy):
    trainer = HookTrainer(policy, TrainerConfig(gradient_accumulation_steps=2))
    batch = collate(samples()[:2])
    trainer.train_step(batch, dry_run=True)
    assert trainer.sliced == 2


def test_ce_chunking_matches_unchunked_loss_and_grads(policy):
    from megatrain import TensorBatch, sft_loss

    batch = collate(samples()[:2])
    tensors = TensorBatch(batch)
    logits = policy(tensors.input_ids)
    full, _ = sft_loss(logits, tensors)
    chunked, _ = sft_loss(logits, tensors, ce_chunk_tokens=2)
    assert float(full.detach().flatten()[0]) == pytest.approx(
        float(chunked.detach().flatten()[0]), abs=1e-6
    )
    full.backward(retain_graph=True)
    full_grads = [p.grad.detach().clone() for p in policy.parameters()]
    policy.zero_grad()
    chunked.backward()
    for expected, actual in zip(full_grads, [p.grad for p in policy.parameters()]):
        assert float((expected - actual).abs().sum()) == pytest.approx(0.0, abs=1e-6)


def test_native_ignore_ce_matches_native_ce(policy):
    from megatrain import TensorBatch, sft_loss

    tensors = TensorBatch(collate(samples()[:2]))
    logits = policy(tensors.input_ids)
    native, _ = sft_loss(logits, tensors, ce_impl="native")
    ignored, _ = sft_loss(logits, tensors, ce_impl="native_ignore")
    assert float(native.detach().flatten()[0]) == pytest.approx(
        float(ignored.detach().flatten()[0]), abs=1e-5
    )


def test_invalid_amp_dtype_rejected():
    with pytest.raises(ValueError, match="amp_dtype"):
        BaseTrainer(TinyPolicy(), TrainerConfig(amp_dtype="fp8"))


def test_logsumexp_ce_matches_native_ce(policy):
    from megatrain import TensorBatch, sft_loss

    tensors = TensorBatch(collate(samples()[:2]))
    logits = policy(tensors.input_ids)
    native, _ = sft_loss(logits, tensors, ce_impl="native")
    fused, _ = sft_loss(logits, tensors, ce_impl="logsumexp")
    assert float(native.detach().flatten()[0]) == pytest.approx(
        float(fused.detach().flatten()[0]), abs=1e-5
    )


def test_device_tensor_batch_path_matches_host_batch_path():
    from megatrain import TensorBatch

    host_batch = collate(samples()[:2])
    tp.manual_seed(1234)
    host_metrics = BaseTrainer(
        TinyPolicy(), TrainerConfig(learning_rate=1e-2)
    ).train_step(host_batch, dry_run=True)
    tp.manual_seed(1234)
    trainer = BaseTrainer(TinyPolicy(), TrainerConfig(learning_rate=1e-2))
    metrics_device = trainer.train_step(TensorBatch(host_batch), dry_run=True)
    assert metrics_device["train/loss"] == pytest.approx(host_metrics["train/loss"], abs=1e-6)


def test_seeded_trainers_are_reproducible():
    def run():
        tp.manual_seed(1234)  # model weights are created before the trainer seeds RNGs
        trainer = BaseTrainer(
            TinyPolicy(), TrainerConfig(learning_rate=1e-2, seed=1234)
        )
        return trainer.train_step(pack_samples(samples(), 8, shuffle=False)[0])

    assert run()["train/loss"] == pytest.approx(run()["train/loss"], abs=1e-7)


def test_compile_policy_respects_dynamic_flag(monkeypatch, policy):
    from megatrain import BaseTrainer, TrainerConfig

    seen = []

    def fake_compile(fn, **kwargs):
        seen.append(kwargs)
        return fn

    monkeypatch.setattr(tp, "compile", fake_compile)
    BaseTrainer(policy, TrainerConfig(compile_policy=True))
    BaseTrainer(policy, TrainerConfig(compile_policy=True, compile_dynamic=True))
    assert [kw["dynamic"] for kw in seen] == [False, True]


def test_compile_policy_off_by_default(policy):
    from megatrain import BaseTrainer, TrainerConfig

    trainer = BaseTrainer(policy, TrainerConfig())
    assert trainer._compiled is None


def test_fused_optimizer_kwargs_are_cuda_only(policy):
    from megatrain import BaseTrainer, TrainerConfig

    cpu = BaseTrainer(policy, TrainerConfig(fused_optimizer=True))
    assert cpu._adam_kwargs() == {}

    cpu._device_type = "cuda"
    assert cpu._adam_kwargs() == {"fused": True, "capturable": True}

    off = BaseTrainer(policy, TrainerConfig())
    off._device_type = "cuda"
    assert off._adam_kwargs() == {}
