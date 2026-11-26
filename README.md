# megaTrain

**TensorPlay 生态的通用训练内核**：把"怎么训练"（优化器、FSDP、编译、梯度累积、AMP、梯度裁剪、checkpoint/恢复）从"训练什么目标"（预训练 CE、SFT、蒸馏、RL 策略梯度）中分离出来。

- `megaTrain` 只依赖 TensorPlay，不知道 rollout、reward、policy version 的存在。
- `megaRL` 等后训练框架依赖 `megaTrain`，在其上实现 RL 算法与异步控制面。
- 这不是玩具实现：前向/反向/优化器/checksum 校验都是真实逻辑；CPU 上的 tiny 模型只用于确定性单元测试。

## 边界

| 层 | 负责 |
| --- | --- |
| megaTrain（本仓库） | 优化器、FSDP/compile、AMP、梯度裁剪、microbatch 累积、打包、checkpoint/RNG/恢复、通用 CE/KL 目标 |
| megaRL | GRPO/PPO/DAPO/蒸馏/DPO 算法实现、陈旧度、奖励、异步控制器、Rust 控制面与权重织物 |
| TensorPlay | 算子、autograd、设备插件 |

## 安装

```bash
python -m pip install -e ".[test]" --extra-index-url https://download.tensorplay.cn/simple
```

## 预训练 / SFT

```python
from megatrain import BaseTrainer, TrainerConfig, TrainSample, collate

samples = [
    TrainSample(token_ids=[...], loss_mask=[0, 0, 1, 1, ...]),  # 1 = 目标 token
]
trainer = BaseTrainer(
    model,
    TrainerConfig(
        device="cuda",
        learning_rate=1e-5,
        gradient_accumulation_steps=4,
        max_grad_norm=1.0,
        amp=True,
        fsdp=True,
        seed=42,
        checkpoint_dir="ckpt",
    ),
    scheduler=warmup_cosine_scheduler,
)
for metrics in trainer.fit(
    samples, max_tokens=8192, alignment=16, epochs=3, checkpoint_every=100
):
    log(metrics)
```

`TrainSample.loss_mask` 标记**目标 token**：prompt/padding 为 0。长文档按 `max_tokens` 和 `alignment`（CP/TP 训练可用 `2 * CP * TP`）切窗后按 token 预算打包，`cu_seqlens` 供 varlen attention 使用。

## 自定义目标

`loss_fn(logits, tensor_batch, **kwargs) -> (loss, diagnostics)`：

```python
from megatrain import BaseTrainer, sft_loss

metrics = trainer.train_step(
    batch,
    loss_fn=sft_loss,
    loss_kwargs={"distill_coef": 0.1, "kl_type": "k3"},
    aux={"teacher_logprobs": teacher_logps},  # 逐行张量按 microbatch 自动切片
)
```

`sft_loss_with_head(hidden, batch, head_weight=...)` 是同一目标、但把 vocab 投影折进 loss 的版本：投影写出的 `[tokens, vocab]` logits 只分配一次，随后的 CE 反向直接把它当作梯度缓冲复用（不再有 fp32 上采样 / `log_softmax` 输出），同结构实测 **1.239×**（bf16 权重）/ 1.233×（fp32 权重 + bf16 autocast）、step 峰值 **−2.0 GiB**，loss 与参考实现一致（[`devdocs/performance.md`](devdocs/performance.md) §4.6）。它需要模型返回 hidden 而不是 logits，其余掩码/归一化/诊断与 `sft_loss` 完全一致。

`sft_loss` 默认是 shifted next-token CE；传入冻结 teacher 的 per-token logprobs 后加上 `coef * KL(student || teacher)`。RL 框架可注入自己的 `policy_loss`，并通过 `tensor_batch_cls` / `_slice_batch` / `_keep_tokens` 钩子携带额外字段和定义归一化单位，而无需复制训练循环。另有 `_needs_auto_teacher`（按目标决定是否自动跑 teacher）与 `checkpoint_artifacts` / `load_checkpoint_artifacts`（把 critic 等额外张量纳入同一 manifest 的 checksum 与裁剪）。

## 恢复

```python
state = trainer.load_checkpoint("ckpt/LATEST")  # 校验 SHA-256 后恢复权重/优化器/scheduler/RNG
```

checkpoint 为原子发布 + `LATEST` 指针 + 按 mtime 裁剪；`extra` sidecar 与任意 manifest 字段（如 RL 侧的 `version`、`reward_stats`）原样保留。

## 生产特性

| 能力 | 位置 |
| --- | --- |
| AdamW / StatelessAdam、可注入优化器 | `trainer.py`, `stateless_adam.py` |
| FSDP `fully_shard` + `no_sync` 累积、`tp.compile`（fixed-shape/dynamic）、fused AdamW | `trainer.py` |
| AMP autocast + GradScaler | `trainer.py` |
| 全局梯度裁剪 + `train/grad_norm` 遥测 | `trainer.py` |
| LR scheduler 逐步 + 状态入 checkpoint | `trainer.py` |
| token-budget 打包、长文档切窗、varlen 边界 | `batch.py` |
| MEGA 权重/优化器序列化、SHA-256、RNG、原子发布 | `trainer.py` |
| RL 侧扩展钩子（batch 类型 / 切片 / 归一化 / teacher / checkpoint artifacts） | `trainer.py` |

## 测试

```bash
python -m pytest tests -q        # 39 个单测（CPU 为主，GPU 项自动跳过），真实张量和优化器
ruff check src tests bench
```

若本机 `import tensorplay` 报 `libnccl.so.2` 缺失，把 pip 包里的 NCCL 库路径加入 `LD_LIBRARY_PATH`（本机为 `/home/bluemoon/miniconda3/lib/python3.13/site-packages/nvidia/nccl/lib`）。

## 性能

训练核心微基准（需 CUDA）：

```bash
python bench/bench_step.py --device cuda --steps 50 --seq 4096 \
  --batch 4 --dim 4096 --vocab 128000 --accum 4 --amp
```

A/B 脚本：`python bench/ab_gpu.py --steps 12 --seq 1024 --batch 4 --dim 1024 --vocab 65536 --accum 2`。

**与 Megatron-Core + Transformer Engine 的同模型 head-to-head**（同 160.46M 参数 GPT、同 bf16、同 token 预算、dropout=0，RTX 3090 单卡，12 step × 3 次中位数；TE 2.21.0 已装并确认走 cuDNN `FusedAttention`）：

```bash
python bench/compare.py --steps 12 --repeat 3 --preset bf16 --min-free-mib 20000
```

表中标着 **TP 本地补丁** 的那一行需要本仓库对 TensorPlay 源码的改动（见 [`upstream-checkouts/UPSTREAMS.md`](upstream-checkouts/UPSTREAMS.md)）；
upstream 1.3.0 的 `linear_cross_entropy` 是参考实现，没有这个收益。

| 栈 | 配置 | tokens/s | step ms | step 峰值 MiB |
|---|---|---:|---:|---:|
| Megatron-Core | bf16 + TE | 66,956 | 61.17 | 2,769 |
| Megatron-Core | **bf16 + TE + FusedAdam（厂商最强配）** | **70,254** | 58.30 | 3,377 |
| TensorPlay | bf16 + 同一套 math attention | 68,589 | 59.72 | 4,647 |
| TensorPlay | bf16 + SDPA | 77,490 | 52.86 | 4,280 |
| TensorPlay | bf16 + SDPA + `tp.compile` + fused AdamW（原生产配置） | 85,124 | 48.12 | 3,070 |
| **TensorPlay** | **bf16 + SDPA + 融合 CE + fused AdamW（当前最佳，TP 本地补丁）** | **97,791** | **41.89** | **2,232** |
| Megatron-Core | fp32 | 27,294 | 150.07 | 4,077 |
| TensorPlay | fp32 + math attention | **31,211** | 131.24 | 5,710 |

- **bf16 行全部来自同一个安静窗口**（`bench/compare.py --preset bf16`，12 step × 3 reps）：
  当前最佳 **1.461×**（vs Megatron+TE）/ **1.392×**（vs Megatron+TE+`FusedAdam`）；原生产配置 1.271× / 1.212×。

本仓库**当前最佳**的 TensorPlay-only 拆解（bf16 权重，2026-11-26 同窗口、3 reps、`bench/fused_ce.sh`；含编译 trunk 的参考 CE 基线）——`compare.py` 整轮复现了其中 97,791 / 97,761 两行：

| TensorPlay 配置（bf16 权重，160.46M） | tok/s | step ms | step 峰值 MiB |
|---|---:|---:|---:|
| 参考 CE + `tp.compile` + fused AdamW（原生产配置） | 84,542 | 48.45 | 3,070 |
| 参考 CE，eager + fused AdamW | 78,264 | 52.33 | 4,280 |
| **融合 CE，eager + fused AdamW（现最佳，推荐）** | **96,935** | 42.25 | **2,232** |
| 参考 CE，编译 trunk + 编译区外 eager loss + fused AdamW | 77,464 | 52.88 | 4,707 |
| 融合 CE + 编译 trunk + fused AdamW | 97,292 | 42.10 | 2,659 |

- **融合 CE（TP 本地补丁，§4.6）**：bf16-权重档（与上表同配置）同窗口 A/B —— 参考 CE 78,264 → 融合 CE **96,935** tok/s = **1.239×**，step 峰值 4,280 → **2,232 MiB（−2.0 GiB）**；编译 trunk 布局下 77,464 → 97,292 = **1.256×**，同样省 2,048 MiB。对原生产配置（84,542，整步 compile）是 **1.147×**（推荐行 96,935）/ **1.151×**（编译 trunk 行 97,292），峰值 3,070 → **2,232 MiB**（推荐行，已低于 Megatron+TE 的 2,769）。loss 与参考一致（11.21086 vs 11.21085）。fp32-权重 + autocast 档同样测得 1.233×，两档一致。
- **`tp.compile` 在融合之后就不划算了**：整步编译 84,542 的增益几乎全来自编译 CE 那几个逐元素算子；CE 融合后，只编译 trunk（77,464）并不比不编译（78,264）快，compile 与不 compile 的融合配置区间重叠（97,292 vs 96,935）却多占 427 MiB。
- **跨栈是同窗口数字**（不再是跨窗口推算）：`bench/compare.py --preset bf16` 把 Megatron 与 TensorPlay 的全部 bf16 行放在一个安静窗口里跑完 —— 融合 CE **97,791** 对 MG+TE 66,956 = **1.461×**、对 MG+TE+FusedAdam 70,254 = **1.392×**；编译 trunk 版 97,761 与之在噪声内。
- 同算术（math attention）vs Megatron+TE+`FusedAdam`：**0.976×** —— 优势来自融合 attention kernel + 图执行，而不是训练循环；fp32 下快 1.144×。
- 显存：原生产路径 3,070 MiB vs TE 2,769 MiB（+301 MiB）；融合 CE 后 **2,659 / 2,232 MiB**（已低于 TE），且同结构对参考 CE 是 **−2.0 GiB**。
- fused AdamW 在 160M 单步内是噪声（±0.4%），保留它是因为厂商最佳实践、且在累积/大 batch 下占比更高。

放大到 474.07M 参数、seq 2048（`--seq 2048 --batch 2 --hidden 2048 --layers 4 --heads 16`）：Megatron+TE 23,555（+`FusedAdam` 24,719）tok/s，TensorPlay SDPA+compile+fused 26,719 tok/s → **1.134× / 1.081×**；同算术是 0.835×。规模越大，融合 kernel 的相对收益越大，但"最强对最强"只剩 8%。**融合 CE** 在 474M 同窗口内是 26,823 → **27,866** tok/s（**1.039×**）、step 峰值 6,703 → **5,607 MiB**（−1.1 GiB）——CE 在 474M 单步里占比更小，所以收益从 160M 的 15.1% 降到 3.9%；跨窗口对 Megatron 则约 1.13× / 1.18×。

**等有效 batch 16**（TensorPlay `batch 4 × accum 4 + compile` vs Megatron `batch 16 + TE + FusedAdam`）：88,251 vs 78,584 tok/s → **1.123×**，而 step 显存 **3,070 vs 8,851 MiB（2.9× 更省）**；累积不复制激活。

推荐的**当前最佳**配置（对应 **96,935–97,791 tok/s / 2,232 MiB** 那两行 —— 同配置、两个安静窗口；数字来自 `bench/tp_gpt_bench.py` 的循环，`BaseTrainer` 另有簿记开销，见 `devdocs/performance.md` §1；需要打过 CE 补丁的 TensorPlay）：

```python
policy = policy.to(tp.bfloat16)          # bf16 权重
cfg = TrainerConfig(
    device="cuda",
    amp=True, amp_dtype="bfloat16",      # 只开 autocast：LN/CE 保持 fp32（与 Megatron 同精度）
    compile_policy=False,                # CE 融合后 compile 只剩噪声且多占 427 MiB
    fused_optimizer=True,                # CUDA only；厂商最佳实践
    gradient_accumulation_steps=1,       # 设 4 可把有效 batch ×4
)
trainer = BaseTrainer(policy, cfg)
# policy(...) 返回 hidden（不接输出头），head 归 loss 管：投影只写一次 logits 缓冲。
trainer.train_step(
    batch,
    loss_fn=sft_loss_with_head,
    loss_kwargs={"head_weight": policy.head.weight},
)
```

若只用 upstream TensorPlay（没有本仓库的 CE 补丁），则回到 `loss_fn=sft_loss` + `compile_policy=True`，
即 85,124 tok/s / 3,070 MiB：融合 CE 的那 15%（1.461×/1.392× 里的一部分）与约 0.8 GiB 都来自那个补丁。

要保留 fp32 master 权重则是 `amp=True` + 不 cast 模型（71,171 tok/s，1.032×）——更稳但更慢，二者别混。

**哪些旋钮真的有用**（160.46M、batch 4、同窗口 3 次中位数，完整矩阵见 [`bench/results/opt_sweep.md`](bench/results/opt_sweep.md)；本表是 **fp32 权重 + bf16 autocast** 档，bf16-权重档见上一节）：

| 旋钮 | tok/s | vs 基线 | 判定 |
|---|---:|---:|---|
| 基线（SDPA，不编译） | 77,299 | — | 参照 |
| `tp.compile(dynamic=False)` | 84,645 | **+9.5%** | ✅ 唯一显著正向 |
| `mode=max-autotune` / `fullgraph=True` | 84,075 / 84,297 | +8.8% / +9.1% | 与 `reduce-overhead` 等价 |
| `compile` + fused AdamW | 84,701 | +9.6% | 噪声内（+0.07%） |
| 整步 CUDA Graph + fused AdamW | 78,377 | +1.4% | ❌ 不如 compile，且与 compile 互斥 |
| `accum=4` + compile | 88,183 | +14.1% | ✅ 但有效 batch ×4 |
| **融合 CE**（`F.linear_cross_entropy`，重写了 TP 源码） | **88,839** | **+14.9%** | ✅ 最大单项；同结构 A/B 1.233×，step 峰值 −2.0 GiB（§4.6） |

- `F.linear_cross_entropy` 在 upstream TensorPlay 1.3 里是**参考实现**（内部仍物化 `[tokens, vocab]` logits），不省显存也不加速。本仓库**改写了 TP 源码**（`tensorplay/nn/_fused_ce.py`，Triton 原地 CE）后它才真的融合：同结构 1.233×、step 峰值 −2.0 GiB、loss 与 fp32 参考一致。它只能 eager 应用（`tp.compile` 捕获会冻结 kernel 的 per-row 缓冲）；而融合之后 compile 本来就只剩噪声（同档 97,292 vs 96,935，且多占 427 MiB），所以本尺寸下 `compile_policy` 可以直接关掉，见 [`devdocs/performance.md`](devdocs/performance.md) §4.6。
- 唯一还能再挤出 ~4% 的是把 CE 从 fp32 降到 bf16（88,056 tok/s，固定 seed 探针 89,192），但它改变 loss 轨迹（10 步后 11.1875 vs 11.2169），而 Megatron 用 fp32 CE，所以**不纳入 headline**。证据：[`bench/results/opt_accum_precision.md`](bench/results/opt_accum_precision.md)。

> 更早版本用"未装 TE"的回退路径做过对比并得到 1.33×；该数据已**作废**，只保留在 [`devdocs/performance.md`](devdocs/performance.md) 的 4.3 附录，用于说明为什么必须先装 TE。TE 安装证据见 [`bench/results/te_install.md`](bench/results/te_install.md)；Bridge/NeMo-RL 在 Python 3.13 装不上的硬原因见 [`bench/results/nvidia_stack_install.md`](bench/results/nvidia_stack_install.md)。

优化项、指标体系与**实测结果**见 [`devdocs/performance.md`](devdocs/performance.md)，原始输出见 [`bench/results/compare_te.md`](bench/results/compare_te.md)。RTX 3090 单卡 A/B：AMP bf16（无 GradScaler）约 2.24× 吞吐、显存 −1.1 GiB；固定 shape 下 `tp.compile(dynamic=False)` 是正优化（+9.5%），`dynamic=True` 才是负优化，已做成 `compile_policy`/`compile_dynamic`；fused AdamW 做成 `fused_optimizer`。对标源码 pin 在 [`upstream-checkouts/UPSTREAMS.md`](upstream-checkouts/UPSTREAMS.md)。融合 CE kernel 已实现并实测（§4.6，本地 TP 补丁）；张量/流水线等 6D 并行尚未实现。

## 与 megaRL 集成

见 [`devdocs/megarl-integration.md`](devdocs/megarl-integration.md)。

## 验证状态

CPU 单元测试覆盖数值正确性、累积一致性、裁剪、scheduler、打包、checksum、恢复与 checkpoint artifacts（39 个，GPU 项在无卡时自动跳过）。GPU 侧做了 **RTX 3090 单卡、同模型/同 dtype/同 token 预算** 的 head-to-head（含已安装的 Megatron-Core + Transformer Engine）；多卡收敛、6D 并行、设备插件与端到端 RL 吞吐尚未验证，不应据此宣称整体性能。融合 CE 的数值与显存结论有独立测试（`tests/test_fused_ce.py`）和 TP 侧 CUDA 测试（`test/nn/test_nn_functional.py`）。
