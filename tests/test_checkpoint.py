"""Checkpoint integrity, generic sidecars and downstream RL fields."""

import math

import pytest
import tensorplay as tp

from megatrain import BaseTrainer, TrainerConfig, TrainSample, collate
from megatrain.trainer import MANIFEST

from .conftest import TinyPolicy


def make_trainer(policy, **kwargs):
    return BaseTrainer(policy, TrainerConfig(learning_rate=1e-2, **kwargs))


def test_checkpoint_roundtrip_preserves_weights_step_and_fields(tmp_path, policy):
    trainer = make_trainer(policy, checkpoint_dir=str(tmp_path))
    batch = collate([TrainSample([1, 2, 3, 4], [0, 0, 1, 1])])
    trainer.train_step(batch)
    path = trainer.save_checkpoint(
        tmp_path / "step-1",
        extra={"dataset": "toy"},
        version=3,
        reward_stats={"mean": 0.5},
    )
    assert path.exists()
    assert (tmp_path / "LATEST").read_text() == "step-1"

    trained = trainer.policy.head.weight.detach().clone()
    fresh = make_trainer(TinyPolicy(), checkpoint_dir=str(tmp_path))
    state = fresh.load_checkpoint(tmp_path / "LATEST")
    assert fresh.step_count == 1
    assert state["version"] == 3
    assert state["reward_stats"] == {"mean": 0.5}
    assert state["extra"] == {"dataset": "toy"}
    assert float((fresh.policy.head.weight.detach() - trained).abs().sum()) == pytest.approx(0.0)


def test_checkpoint_detects_corruption(tmp_path, policy):
    trainer = make_trainer(policy)
    path = trainer.save_checkpoint(tmp_path / "v1")
    (path / "model.mega").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="checksum mismatch"):
        make_trainer(TinyPolicy()).load_checkpoint(path)


def test_checkpoint_pruning_and_idempotent_save(tmp_path, policy):
    trainer = make_trainer(policy, checkpoint_dir=str(tmp_path), max_checkpoints=2)
    for index in range(3):
        trainer.save_checkpoint(tmp_path / f"v{index}", version=index)
    assert (tmp_path / "v0").exists() is False
    assert (tmp_path / "v1").exists()
    assert (tmp_path / "v2").exists()
    assert (tmp_path / MANIFEST).exists() is False
    path = trainer.save_checkpoint(tmp_path / "v2", version=999)
    assert path.name == "v2"
    assert (tmp_path / "v2" / MANIFEST).exists()


class ArtifactTrainer(BaseTrainer):
    """Downstream RL trainer stand-in: a critic checkpointed with the policy."""

    def __init__(self, policy, config, critic):
        super().__init__(policy, config)
        self.critic = critic
        self.restored = None

    def checkpoint_artifacts(self):
        return {"value.mega": self.critic.state_dict()}

    def load_checkpoint_artifacts(self, root, files):
        assert "value.mega" in files
        self.critic.load_state_dict(tp.load(root / "value.mega"))


def test_checkpoint_artifacts_are_hashed_and_restored(tmp_path):
    from .conftest import TinyPolicy

    critic = TinyPolicy()
    trainer = ArtifactTrainer(TinyPolicy(), TrainerConfig(), critic)
    batch = collate([TrainSample([1, 2, 3, 4], [0, 0, 1, 1])])
    trainer.train_step(batch)
    path = trainer.save_checkpoint(tmp_path / "v1", version=1)
    assert (path / "value.mega").exists()
    import json

    manifest = json.loads((path / MANIFEST).read_text())
    assert "value.mega" in manifest["files"]

    fresh_critic = TinyPolicy()
    fresh = ArtifactTrainer(TinyPolicy(), TrainerConfig(), fresh_critic)
    fresh.load_checkpoint(path)
    assert all(
        math.isclose(a, b, abs_tol=0.0)
        for a, b in zip(
            fresh_critic.head.weight.detach().flatten().tolist(),
            critic.head.weight.detach().flatten().tolist(),
        )
    )
