# megaRL → megaTrain 接入方案

状态：**已落地**（megaRL `src/megarl/trainer.py` 的 `TensorPlayTrainer`
直接继承 `megatrain.BaseTrainer`；`kl.py` / `stateless_adam.py` 已改为
re-export）。下方为文件级映射与验证约定。

目标：megaRL 只保留 RL 算法与编排；训练内核、checkpoint、通用目标来自 `megatrain`。控制器侧无需改动。

## 1. 依赖与发布顺序

1. 发布 `megatrain>=0.1,<0.2`。
2. megaRL 的 `pyproject.toml` 加入 `megatrain>=0.1,<0.2`。
3. megaRL 删除或改为 re-export 的模块：`stateless_adam.py`、`kl.py`。其余算法模块保持不动。

## 2. 文件级映射

| megaRL 现状 | 接入 megaTrain 后 |
|---|---|
| `trainer.py::TrainerConfig` 的通用字段（device/lr/opt/accum/fsdp/compile/ckpt） | 用 `megatrain.TrainerConfig` 组合，RL 字段留在 megaRL 自己的 config |
| `trainer.py::TensorPlayTrainer` 的优化器/FSDP/compile/累积/ckpt 代码 | 继承 `megatrain.BaseTrainer`，删掉重复实现 |
| `losses.py::policy_loss` 与 GRPO/PPO/DAPO/... | 留在 megaRL，作为注入的 `loss_fn` |
| `dpo.py` | 留在 megaRL；`dpo_update` 作为 RL 子类的独立方法 |
| `advantages.py`, `corrections.py` | 留在 megaRL（RL 语义） |
| `batching.py::RLBatch` | 保留 RL 字段；通过钩子适配 `megatrain.TrainBatch` |
| `controller.py` / `async_controller.py` | **不改**：它们通过 `set_version` / `update` / `dpo_update` duck typing 调用 trainer |

## 3. RLTrainer 骨架

```python
from megatrain import BaseTrainer, TrainerConfig as CoreConfig
from megatrain.losses import TensorBatch as CoreTensorBatch

class RLTensorBatch(CoreTensorBatch):
    def __init__(self, batch, device="cpu"):
        super().__init__(batch, device=device)
        self.old_logprobs = tp.tensor(batch.old_logprobs, dtype=tp.float32, device=device)
        self.versions = tp.tensor(batch.versions, dtype=tp.long, device=device)
        self.sampling_logprobs = ...  # 同现有实现

class TensorPlayTrainer(BaseTrainer):
    tensor_batch_cls = RLTensorBatch

    def __init__(self, policy, config, **kwargs):
        core = CoreConfig(
            device=config.device,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            optimizer_kind=config.optimizer_kind,
            learning_rate=config.learning_rate,
            # ...
        )
        super().__init__(policy, core, **kwargs)
        self.rl_config = config  # loss_type, KL, 陈旧度, 优势估计等
        self.version = 0
        self.adaptive_kl = ...

    def set_version(self, version):
        self.version = int(version)

    def _slice_batch(self, batch, start, stop):
        # 用现有 _slice_batch 实现，保留 old_logprobs/versions/rewards 等字段
        return slice_rl_batch(batch, start, stop)

    def _keep_tokens(self, batch):
        # 保留现有 _micro_weight / _global_token_count 的归一化语义
        return count_fresh_valid_tokens(batch, self.version, self.rl_config.max_offpolicy_tokens)

    def update(self, batch, advantages, *, clip_ratio=0.2, kl_coef=0.0, dry_run=False):
        # 从现有 update() 迁移：准备 teacher/ref/value/GAE 的 aux 张量，
        # 然后调用 train_step(batch, loss_fn=policy_loss, loss_kwargs={...}, aux={...})
        return self.train_step(
            batch,
            loss_fn=policy_loss,
            loss_kwargs={"epsilon_low": ..., "kl_coef": ..., "advantages_per_row": ...},
            aux=self._rl_aux(batch),
            dry_run=dry_run,
        )
```

注意：旧版 `forward_fn(model, ids)` 的签名变为 `forward_fn(model, tensor_batch)`；已有视觉/多模态前向需改成从 `tensor_batch.input_ids` 等字段取值。

## 4. 优势/陈旧度数据的传递

`train_step` 的 `aux` 字典会按 batch 第 0 维自动切给每个 microbatch，因此：

- `advantages`（逐行或逐 token）→ `aux={"advantages": adv_tensor}`
- `ref_logprobs` / `teacher_logprobs` / `old_logprobs` → 同样通过 `aux`
- PPO 的 `values` / `returns` → 同样通过 `aux`
- `policy_loss` 的标量配置通过 `loss_kwargs` 传，避免与逐行张量混淆。

## 5. checkpoint 兼容

megaTrain 的 manifest 把未知字段原样写在顶层：

```python
trainer.save_checkpoint(
    path, version=self.version, accepted_tokens=..., reward_stats=...
)
state = trainer.load_checkpoint(path)  # state["version"] 等原样返回
```

RL 子类通过 `checkpoint_artifacts()` / `load_checkpoint_artifacts()` 把 PPO
critic（`value.mega`）纳入同一 manifest 的 checksum 校验；adaptive-KL 状态以
manifest 顶层 `kl` 字段保存。checkpoint 只使用 `model.mega` / `optimizer.mega`
/ `scheduler.mega` / artifact 格式，不保留任何 pre-release 的 `optimizer.pkl`
回退路径。

## 6. 归一化与批内权重

`train_step` 用 `keep_i / sum(keep)` 给 microbatch 加权。RL 子类把归一化族映射为：

| RL 损失族 | `_keep_tokens` 单位 | 结果 |
|---|---|---|
| GRPO / SAPO / LUSPO / Dr.GRPO | 行数 | 按行均值的 batch 均值 |
| BNPO / DAPO / CISPO / VESPO / PPO / reverse-KL / OPD | fresh valid token 数 | 与全局 token 分母逐位等价，且每个 microbatch 图独立 |

PPO 的 value loss 不再跨 microbatch 共享一张图：GAE 用 `no_grad` 的整批
value 估计，value loss 由注入的 `loss_fn` 在当前 microbatch 重新前向。

## 7. 回归验证

1. 先跑 megaRL 现有 `tests/test_trainer.py`、`test_trainer_algorithms.py`、`test_distillation_and_dpo.py` 全绿（签名适配后）。
2. 再跑控制器测试：controller/async_controller 对 trainer 的 duck typing 调用未变。
3. GPU 阶段另行验证 FSDP/compile/AMP 路径；本仓库 CPU 单测不覆盖多卡与设备插件。
