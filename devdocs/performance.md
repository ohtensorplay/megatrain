# 生产优化：实测结论与方法

## 0. 先定义"击败 NVIDIA"

没有同硬件/同模型/同精度的测量，"击败"没有意义。公开基线（工作负载不同，**不可直接对比**）：

- Megatron-Bridge：DGX-H100、64 GPU、BF16、Qwen3-30B-A3B、seq 4096 → 5,984 tokens/s/GPU（[官方表](https://docs.nvidia.com/nemo/megatron-bridge/0.4.1/performance-summary.html)）
- NeMo-RL：GRPO on DeepSeek V3，256 GPU → 111 tokens/s/GPU（[官方表](https://docs.nvidia.com/nemo/rl/nightly/about/performance-summary.html)）
- 第三方 AsyncGRPO 对比里 NeMo-RL 的 useful tokens/s 为 12.2k，TRL/prime-rl 为 53.5k（[来源](https://huggingface.co/spaces/trl-lib/asyncgrpo-benchmark)）：编排层是可攻短板。

## 1. 本机实测 A/B（RTX 3090，TensorPlay 1.3.0.dev，单卡）

复现命令：

```bash
python bench/ab_gpu.py --steps 12 --seq 1024 --batch 4 --dim 1024 \
    --vocab 65536 --accum 2 --chunk 512
```

同一模型/权重/种子/batch，warmup 2 step，CUDA 同步计时；非 AMP 路径的 loss 全部为 10.96258（断言通过）。不同轮次间有约 5–10% 波动，只看量级差异。

| 路径 | tokens/s | step ms | 峰值显存 MiB | vs legacy |
|---|---:|---:|---:|---:|
| legacy（全量 `log_softmax` CE） | ~34,300 | ~119 | 4,665 | 1.00× |
| 零拷贝 / 原生 CE | ~35,400 | ~116 | 4,665 | 1.03× |
| 分块原生 CE（chunk=512） | ~28,800 | ~142 | 4,791 | 0.84× |
| logsumexp CE | ~32,100 | ~128 | 5,175 | 0.94× |
| AMP fp16 + GradScaler | ~75,000 | ~54 | 3,505 | 2.17× |
| **AMP bf16（无 GradScaler）** | **~77,500** | **~53** | **3,505** | **2.24×** |
| AMP bf16 + `native_ignore` CE | ~77,600 | ~53 | 3,505 | 2.24× |
| AMP bf16 + `tp.compile(dynamic=True)`（toy policy） | ~75,200 | ~54 | 5,114 | 2.17× |

### 实测结论

- **AMP 是唯一的大幅收益**：fp16 2.17×，bf16 2.24×，同时显存 −1,160 MiB。bf16 因免去 scale/unscale，比 fp16 再快约 3%，是 Ampere+ 上的推荐默认。
- **被证伪的假设**：
  - logsumexp CE 不省显存也不快，峰值反而 +510 MiB；
  - 分块 CE（512）不省显存也不快，峰值 +126 MiB、吞吐 −16%；
  - `native_ignore`（把 mask 融入 `ignore_index`）在噪声范围内，不是性能杠杆；
  - `tp.compile(..., dynamic=True, mode="reduce-overhead")` 在这个微基准上慢约 3%、峰值显存 +1.6 GiB（动态 shape/图重写开销），不应默认开启。**但这条结论只对 `dynamic=True` 成立**：在真实 GPT（160.46M、seq 1024）上 `dynamic=False` 是正优化 —— 84,116 tok/s、3,070 MiB（§4.4），对比不编译的 77,324 tok/s、4,280 MiB；`dynamic=True` 则掉到 74,892 tok/s、6,826.8 MiB（Stax lowering 失败，退回 graph executor）。因此 `TrainerConfig` 新增 `compile_dynamic=False`。
- 零拷贝路径收益约 3%，主要价值是对接数据面，而不是吞吐。

> `token_logprobs` 的 logsumexp 改写（蒸馏/RL 用）尚未单独做 GPU A/B，不能沿用 CE 的结论宣称收益。

## 2. 还没做、且决定上限的部分

对标源码的版本与 commit 记录在本机私有笔记 `devdocs/local/`（不入库）：Megatron-Bridge、NeMo-RL、Megatron-LM、torchtitan。

NVIDIA 的优势在底层 kernel 与并行，不在 Python 循环：

1. 融合算子（Transformer Engine fused attention/CE/AdamW）；当前 TP CE 是 `log_softmax + nll_loss`。
2. 多维并行 TP/PP/CP/EP、MoE expert parallel、pipeline 负载均衡；megaTrain 目前只有 FSDP。
3. 整步 CUDA Graph：实测只 +1.6%，明显不如 `tp.compile`，且二者互斥（§4.4），不采用。
4. FP8 训练及缩放策略。
5. 通信/重叠：optimizer 与梯度通信重叠、权重同步与生成重叠。
6. 数据管线（tokenizer、预 tokenize 存储、shuffle/恢复）；本仓库只接预 tokenize 的 `TrainSample`。
7. 分布式 sharded checkpoint；目前存完整 state_dict。

## 3. 测量协议

训练核心用 `bench/bench_step.py` / `bench/ab_gpu.py`；RL 端到端必须分别报告生成、奖励、训练、权重同步四段耗时，并固定 GPU 型号/数量、模型、精度、seq、GBS/MBS、warmup ≥2 step，报 `tokens/s/GPU`、`step_ms`、`peak_memory_bytes`。

所有数字都来自 **RTX 3090 单卡训练核心微基准**，不含生成和奖励，不能与 NeMo-RL/verl 的 E2E 数字直接比较。

## 4. 同模型 head-to-head：Megatron-Core + Transformer Engine（有效证据）

> **旧数据作废声明**：本节之前用"TE 未安装"的 Megatron torch 回退路径做过对比并得到 1.33×，那不代表 NVIDIA 的生产栈，已移入 [4.3 附录](#43-附录未装-te-的数据无效仅存档)。以下数字全部在 **TE 2.21.0 装好、确认走 cuDNN FusedAttention 之后**测得。

- 结构：vocab 65536、hidden 1024、2 层、8 头、seq 1024、micro-batch 4；learned absolute position embedding、pre-norm、GELU 4× MLP、未共享 embedding/输出头、全 token 平均 CE、dropout 0。
- 两边都报告 **160.46M 参数**（`tests/test_bench_architecture.py` 用解析公式锁死等价性）。
- 两边跑的是各自的**同一个手写训练循环**（`bench/tp_gpt_bench.py` / `bench/megatron_bench.py`：同样的模块、同一次 forward/backward/step、同样的计时点），
  不是 `BaseTrainer` 的簿记路径；`BaseTrainer` 自身的对照在 §1（`bench/bench_step.py`）。
- 计时：warmup 2 step，计时 12 step，`cuda.synchronize()` 前后取值，跑 3 次取中位数。这台 3090 与另一个会话共用，所以 `bench/compare.py` 每次计时前都要求 GPU `util=0` 且空闲显存达标（默认 20 GiB，本文用 16 GiB），计时窗口被租户扰动时丢弃重测。
- NVIDIA 侧：`megatron-core 0.16.1` + `transformer_engine 2.21.0`（cp313 源码编译的 torch 绑定）、cuBLAS 13.8.2.12；`NVTE_DEBUG` 确认 `Selected backend = FusedAttention (sub-backend 1)`，`te_layers` 含 TE 的 `LayerNorm`。证据：[`../bench/results/te_install.md`](../bench/results/te_install.md)。
- Megatron 的 `attention_dropout` / `hidden_dropout` 默认 0.1，基准里显式设为 0，否则等于让 NVIDIA 侧多算 dropout。

复现：

```bash
python bench/compare.py --steps 12 --repeat 3
```

| 栈 | 配置 | 参数 M | tokens/s | step ms | step 峰值 MiB | 波动 |
|---|---|---:|---:|---:|---:|---:|
| Megatron-Core | bf16 + TE（融合 LN/attention） | 160.46 | 66,956 | 61.17 | 2,769 | ±0.6% |
| Megatron-Core | **bf16 + TE + FusedAdam（厂商最强配）** | 160.46 | **70,254** | 58.30 | 3,377 | ±0.3% |
| Megatron-Core | bf16 + 未融合 math attention | 160.46 | 59,706 | 68.60 | 3,261 | ±0.5% |
| TensorPlay | bf16 + math attention（同一套算术） | 160.46 | 68,589 | 59.72 | 4,647 | ±0.1% |
| TensorPlay | bf16 + SDPA attention | 160.46 | 77,490 | 52.86 | 4,280 | ±0.7% |
| TensorPlay | bf16 + SDPA + `tp.compile(dynamic=False)` | 160.46 | 84,288 | 48.60 | 3,070 | ±1.0% |
| TensorPlay | bf16 + SDPA + compile + `fused AdamW`（原生产配置） | 160.46 | 85,124 | 48.12 | 3,070 | ±0.9% |
| **TensorPlay（TP 补丁）** | **bf16 + SDPA + 融合 CE + `fused AdamW`（当前最佳）** | 160.46 | **97,791** | **41.89** | **2,232** | ±0.9% |
| TensorPlay（TP 补丁） | bf16 + SDPA + 编译 trunk + 融合 CE + `fused AdamW` | 160.46 | 97,761 | 41.90 | 2,659 | ±1.0% |
| TensorPlay | fp32 权重 + bf16 AMP + SDPA（上一轮窗口） | 160.46 | 71,171 | 57.55 | 5,414 | ±0.9% |
| Megatron-Core | fp32（上一轮窗口） | 160.46 | 27,294 | 150.07 | 4,077 | ±0.6% |
| TensorPlay | fp32 + math attention（上一轮窗口） | 160.46 | 31,211 | 131.24 | 5,710 | ±0.2% |

| 对比（同 dtype、同结构、同 token 预算；bf16 行同一窗口） | 吞吐比 |
|---|---:|
| **融合 CE（当前最佳）vs Megatron bf16 + TE + FusedAdam** | **1.392×** |
| **融合 CE（当前最佳）vs Megatron bf16 + TE** | **1.461×** |
| 融合 CE + 编译 trunk vs Megatron bf16 + TE + FusedAdam | 1.392× |
| 融合 CE + 编译 trunk vs Megatron bf16 + TE | 1.460× |
| 原生产配置（compile + fused AdamW）vs Megatron bf16 + TE + FusedAdam | **1.212×** |
| 原生产配置（compile + fused AdamW）vs Megatron bf16 + TE | **1.271×** |
| TensorPlay bf16 SDPA（不编译） vs Megatron bf16 + TE + FusedAdam | 1.103× |
| TensorPlay bf16 SDPA（不编译） vs Megatron bf16 + TE | 1.157× |
| TensorPlay bf16 math vs Megatron bf16 + TE + FusedAdam | **0.976×** |
| TensorPlay bf16 math vs Megatron bf16 + TE | 1.024× |
| TensorPlay bf16 math vs Megatron bf16 未融合 math | 1.149× |
| TensorPlay fp32 vs Megatron fp32 | 1.144× |

原始输出：[`../bench/results/compare_te.md`](../bench/results/compare_te.md)、[`../bench/results/compare_te.json`](../bench/results/compare_te.json)。
融合 CE 那几行**不在**这两个文件里（它们来自 `bench/results/fused_ce.md` 的同窗口测量，§4.6）。

### 怎么读这些数字（含对旧结论的修正）

1. **最强对最强是 1.21×（原生产配置）/ 1.39×（融合 CE）**。装上 TE 并给 Megatron 配它自己的 `FusedAdam` 后，
   原生产路径（SDPA + 固定 shape 编译 + fused AdamW）85,124 tok/s vs 70,254 tok/s，快 **21.2%**（对 TE 默认 27.1%）；
   加上融合 CE 是 **97,791 vs 70,254 = 1.392×**（对 TE 默认 1.461×）。全部 bf16 行来自**同一个安静窗口**。
2. **优势来自融合 kernel + 图执行，不来自训练循环**：TensorPlay 的 math 路径对 Megatron TE+FusedAdam 是 **0.976×**（慢 2.4%），
   对 TE 默认 1.024×。把这点写清楚，比喊"击败"重要。
3. **`tp.compile` 的结论要修正（并再修正一次）**：`dynamic=True` 是负优化（74,892 tok/s、6,826.8 MiB，Stax lowering 失败退回 graph executor），
   固定 shape 下 `dynamic=False` 是正优化：77,490 → 84,288 = +8.8%，step 峰值 3,070 vs 4,280 MiB（−1.2 GiB）。
   但**它编译的几乎都是 CE 那几个逐元素算子**：把 CE 换成融合实现后，compile 只剩噪声（97,791 vs 97,761）且多占 427 MiB，
   所以当前推荐关掉它（§4.6）。`TrainerConfig` 仍保留 `compile_policy` / `compile_dynamic`（默认 False）。
4. **显存：原生产配置 3,070 MiB vs Megatron+TE 2,769（+301）；融合 CE 后 2,232 MiB，反而比 TE 少 537 MiB**。
   那 301 MiB 的差距本来就是全量 `cross_entropy` 的 workspace；分块 CE 也能把未编译路径的 4,647 降到 4,199（chunk=512）
   但吞吐 −32%，已被融合 CE 取代（同时更快且更省）。
5. **fp32 下快 14.4%**：TE 没有 fp32 融合 attention，两边都退回 math 路径，此时 TensorPlay 的循环/CE 开销更低。
6. **复现 Megatron 必须绕开两个坑**：`bf16=True` 不会改 `params_dtype`（默认 fp32），会造成 float32 线性层 + bf16 softmax 概率在 `bmm` 处 dtype 崩溃；未装 TE 时激活还是 fp32，需要外层 `torch.autocast(bf16)`。两者都已在 `bench/megatron_bench.py` 显式处理。
7. **未给 NVIDIA 侧上的手段**：CUDA Graph 捕获、torch.compile、TP/PP/CP 并行、FP8。这些在单卡 3090 上不可用或不适用，但意味着本结论不能外推到多卡/FP8。

### 4.1 474.07M 参数、seq 2048（同样已装 TE）

```bash
python bench/compare.py --steps 8 --repeat 3 --preset bf16 \
    --seq 2048 --batch 2 --hidden 2048 --layers 4 --heads 16 --vocab 65536
```

| 栈 | 配置 | 参数 M | tokens/s | step ms | 峰值 MiB | 波动 |
|---|---|---:|---:|---:|---:|---:|
| Megatron-Core | bf16 + TE | 474.07 | 23,555 | 174.9 | 5,382 | ±0.2% |
| Megatron-Core | **bf16 + TE + FusedAdam** | 474.07 | **24,719** | 166.7 | 7,175 | ±0.2% |
| Megatron-Core | bf16 + 未融合 math | 474.07 | 17,454 | 235.0 | 8,930 | ±0.1% |
| TensorPlay | bf16 + math | 474.07 | 19,668 | 208.26 | 9,917 | ±0.2% |
| TensorPlay | bf16 + SDPA | 474.07 | 26,447 | 154.88 | 6,910 | ±0.2% |
| TensorPlay | bf16 + SDPA + compile | 474.07 | 26,558 | 154.23 | 6,707 | ±0.1% |
| TensorPlay | bf16 + SDPA + compile + fused AdamW | 474.07 | 26,719 | 153.30 | 6,707 | ±0.7% |
| **TensorPlay（TP 补丁）** | **bf16 + SDPA + 融合 CE + 编译 trunk + fused AdamW** | 474.07 | **27,866** | 147.0 | **5,607** | ±0.4% |

| 对比 | 吞吐比 |
|---|---:|
| TensorPlay 生产配置 vs Megatron bf16 + TE + FusedAdam | **1.081×** |
| TensorPlay 生产配置 vs Megatron bf16 + TE | **1.134×** |
| **融合 CE 行** vs Megatron bf16 + TE + FusedAdam | **1.127×** |
| **融合 CE 行** vs Megatron bf16 + TE | **1.183×** |
| TensorPlay bf16 math vs Megatron bf16 + TE | **0.835×** |

> 474M 的融合 CE 行是 **2 个干净 rep 的均值**（27,783 / 27,949），与上面的 Megatron 行不是同一窗口
> （Megatron 数字来自 TE 有效窗口），所以那两行倍数标 provisional。同窗口内可比的是
> 26,823 → 27,866 = **1.039×**、峰值 6,703 → 5,607 MiB（同样是 2–3 个 rep，恰好逢邻居 util 57–100%，
> 但两次融合 rep 只差 0.6%，且这个几何是算力瓶颈，故未丢弃）。
> CE 在 474M 单步里占比更小，所以收益从 160M 的 15% 降到 3.9%。

- 规模变大后融合 attention 的相对收益更大：Megatron+TE 比自己未融合快 35%，TensorPlay SDPA 比自己的 math 快 34%。
- **最强对最强只剩 8%**（26,719 vs 24,719）；同算术下 TensorPlay 落后 16%（19,668 vs 23,555）。compile 在这个尺寸只多 0.4%（噪声内），但 step 峰值降 203 MiB。
- 显存：TensorPlay 6,707 vs TE 5,382（+1.3 GiB），但仍低于 Megatron 的未融合路径 8,930；融合 CE 后是
  **5,607 MiB（对 TE +225 MiB）**，同结构比参考 CE 省 1,096 MiB。
- **474M 上融合 CE 是 +3.9%，不是 160M 的 +15%**：trunk 大 4 倍而 head+CE 的尺寸不变，
  所以同一份 CE 优化在整步里的占比按几何缩小。方向符合预期。

原始输出：[`../bench/results/compare_te_large.md`](../bench/results/compare_te_large.md)；融合 CE 行来自
[`../bench/results/fused_ce.md`](../bench/results/fused_ce.md) §3。

### 4.2 安装状态（"必须安装"的执行结果）

- `megatron-core==0.16.1`：**已安装**，基准的 NVIDIA 侧框架。
- **Transformer Engine 2.21.0：已安装并跑通**。
  - `transformer_engine` / `transformer_engine_cu13` 用官方 cu13 wheel；
  - `transformer_engine_torch` 官方没有 cp313 wheel，**在本机源码编译成功**（34 个目标文件，`MAX_JOBS=4`，产物 `cp313` wheel）；
  - 原 `nvidia-cublas==13.1.1.3` 不导出 TE core 库需要的 `cublasLtGroupedMatrixLayoutInit_internal`，升级到 `13.8.2.12` 后导入成功，torch 2.13 的 matmul 也正常；
  - `NVTE_DEBUG` 确认 Megatron `--te` 走的是 **cuDNN `FusedAttention`（sub-backend 1）**，不是 torch 回退。
  - 证据与命令：[`../bench/results/te_install.md`](../bench/results/te_install.md)。
- `megatron-bridge`：**装不上，且不是编译问题** —— 所有 ≥0.4 版本 `Requires-Python >=3.12,<3.13`，本机 Python 3.13 被 pip 直接拒绝；唯一可选的 0.3.1 又要求 `transformers<5`、`transformer-engine<2.13`、`nvidia-resiliency-ext`、`mamba-ssm`，与已跑通的栈冲突。证据：[`../bench/results/nvidia_stack_install.md`](../bench/results/nvidia_stack_install.md)。
- `nemo-rl`：PyPI 仅占位 `0.0.0`，官方依赖同样要求 Python 3.12；源码已 clone 作参考，**未安装**。

**注意**：`bench/megatron_bench.py` 内置了半装 TE 的自动回退（把不可导入的 TE 从 `sys.modules` 毒化，强制走 torch 路径），所以坏掉的 TE 不会污染基线；装了 TE 时会打印实际层类型（`te_layers`）。

### 4.3 附录：未装 TE 的数据（无效，仅存档）

最早那版数据**在 Transformer Engine 未安装时**测得，Megatron 走 torch 回退（math attention + Torch LayerNorm）。它不能用于任何"击败 NVIDIA"的结论，只用来解释为什么必须先装 TE：

| 栈 | 配置 | tokens/s | 峰值 MiB |
|---|---|---:|---:|
| Megatron-Core（无 TE） | bf16 unfused | 58,507 | 3,361 |
| TensorPlay | bf16 math | 68,234 | 4,647 |
| TensorPlay | bf16 SDPA | 77,662 | 4,280 |
| Megatron-Core（无 TE） | fp32 | 27,038 | 4,417 |
| TensorPlay | fp32 math | 30,950 | 5,710 |

当时得出的 "1.17× / 1.33×" 全部作废；装上 TE 后对应数字是 0.984× / 1.221×（对厂商最强配）。

### 4.4 极致优化：旋钮矩阵（同一几何，逐项实测）

在 160.46M / seq 1024 / batch 4 上把每个候选旋钮单独打开，同一安静窗口内各跑 3 次取中位数。测量脚本 `bench/opt_sweep.sh` 会等到 GPU `util=0` 且 `free≥18 GiB` 才计时，并丢弃被同机其他租户扰动的窗口（这台机器与另一个会话共用 3090）。

| 旋钮（都基于 SDPA） | tok/s | vs 基线 | step ms | step 峰值 MiB | 进程峰值 MiB | 判定 |
|---|---:|---:|---:|---:|---:|---|
| 基线（无编译） | 77,299 | — | 52.99 | 4,280 | 4,280 | 参照 |
| `tp.compile(dynamic=False, mode=reduce-overhead)` | 84,645 | **+9.5%** | 48.39 | 3,070 | 4,606 | ✅ 已作生产开关 |
| 同上 + `mode=max-autotune` | 84,075 | +8.8% | 48.72 | 3,070 | 4,606 | 无额外收益 |
| 同上 + `fullgraph=True` | 84,297 | +9.1% | 48.59 | 3,070 | 4,606 | 无额外收益 |
| 同上 + `fused AdamW` | 84,701 | +9.6% | 48.36 | 3,070 | 4,606 | 噪声内（+0.07%） |
| 同上 + `F.linear_cross_entropy`（未打补丁的 TP 1.3） | 83,883 | +8.5% | 48.83 | 3,070 | 4,606 | 参考实现，无收益（现已融合，见 §4.6） |
| 整步 CUDA Graph + fused AdamW | 78,377 | +1.4% | 52.26 | 1,225 | 4,280 | ❌ 不如 compile；workspace 只是搬进常驻 graph pool |
| **`accum=4` + compile** | **88,183** | **+14.1%** | 185.80 | 3,070 | 4,489 | ✅ 但有效 batch ×4 |
| `accum=4`（不编译） | 80,139 | +3.7% | 204.45 | 4,586 | 4,586 | 验证 optimizer 摊薄 |
| 无 autocast + fp32 CE + compile | 84,062 | +8.7% | 48.73 | 3,070 | 5,513 | 同精度，无收益；fp32 logits 反而多占显存 |
| 无 autocast + compile（bf16 LN/CE） | 88,056 | +13.9% | 46.52 | 2,558 | 3,334 | ⚠️ 精度降级，不作默认 |

读法：

1. **编译类旋钮里只有 `tp.compile(dynamic=False)` 有价值**：+9.5%，step 快 4.6 ms（52.99 → 48.39）且 step 峰值省 1.2 GiB；`max-autotune`、`fullgraph` 与 `reduce-overhead` 等价，不值得再加复杂度。但**它编译的其实主要是那三个逐元素 CE 算子**：只编译 trunk 时 72,085 ≈ 不编译 72,029，而含 CE 的整步编译 77,934（§4.6）。所以真正的杠杆是 CE 本身——换成融合实现后 88,839 tok/s（同结构 1.233×、step 峰值 −2.0 GiB），编译退化成噪声。
2. **整步 CUDA Graph 是负优化**：它能省掉 Python/launch 间隙，但把 `tp.compile` 的 kernel 融合挡在门外，只比不编译快 1.4%。注意它的 step 峰值 1,225 MiB 是假象 —— workspace 被搬进常驻的 graph pool，进程峰值仍是 4,280 MiB，没有真省。而且 `compile + CUDAGraph` 会嵌套捕获失败（CUDA error 901），二者不可叠加。
3. **fused AdamW 在本尺寸是噪声**（+0.07%，step 2.83 → 2.74 ms）。仍保留为 `TrainerConfig.fused_optimizer`，因为它是厂商最佳实践、且在 accumulation 放大后占比更高；不作为默认。
4. **梯度累积是第二大杠杆**：4 个 micro-batch 摊薄一次 optimizer/zero_grad，88,183 vs 84,645 = +4.2%（相对 compile）。它改变有效 batch（4→16），所以不能算作同 token 预算的 A/B，只能作为吞吐旋钮报告。
5. **`F.linear_cross_entropy` 在 upstream 1.3 不是融合算子**：它照抄参考实现，物化 `[tokens, vocab]` logits 再调 `cross_entropy`（实测 83,883 vs 84,645，噪声内、显存不变）。本仓库把 TP 源码里的 CE 换成真正的融合实现后，这个旋钮变成**最大单项收益**：同结构 A/B **1.233×**、step 峰值 **−2.0 GiB**（§4.6）。
6. **分块 CE / logsumexp CE 不是显存旋钮**：分块 CE 把 workspace 换小但吞吐 −10%~−32%（§3.3），logsumexp 更差；它们想要省的显存，融合 CE 是**免费**拿到的（不需要分块，也不需要 bf16 降精度）。
7. **等有效 batch 16 的对照**：TensorPlay `batch 4 × accum 4 + compile` = 88,251 tok/s、step 峰值 **3,070 MiB**；Megatron `batch 16 + TE + FusedAdam` = 78,584 tok/s、**8,851 MiB**。比值 **1.123×**，但显存是 **2.9×** 优势。累积不复制激活，这是它比"直接开大 batch"更好的原因。
8. **`--no-autocast` 的 +13.9% 不是免费的**：关掉 autocast 后 LayerNorm/CE 从 fp32 降到 bf16，88,056 tok/s、step 2,558 MiB，但 10 步后 loss 11.1875 vs 11.2169。**保持 CE 为 fp32** 时（`--ce-fp32`）回到 84,062 tok/s（进程峰值还因显式 fp32 logits 涨到 5,513 MiB）—— 那多出来的 ~4% 全部来自 CE 精度降级。独立窗口、固定 seed 的探针测得 89,192 / 84,301，与此处 88,056 / 84,062 的差异是运行间波动，结论一致。Megatron 的 `compute_language_model_loss` 会显式把 logits 上采样到 fp32，所以 bf16-CE 不能拿来和它做同精度比较；该旋钮只作为"精度换速度"记录，不进默认路径。证据：[`../bench/results/opt_accum_precision.md`](../bench/results/opt_accum_precision.md)。
9. **不要用 `--no-autocast` 冒充优化**：去掉 autocast 包装本身没有收益（84,062 ≈ 84,645，差 −0.7%），省下的 dispatch 被显式 `logits.float()` 抵消。

> 显存口径：两列都保留，因为单看任何一个都会被误导。`step MiB` 是构建参数/优化器并跑 2 个 warmup step 后 reset、再取计时窗口峰值（激活 + logits workspace + 梯度缓冲），不含常驻参数/优化器状态；`total MiB` 是进程级高水位，含 compile scratch / CUDA graph pool 等常驻块。§4 与 §4.1 的 head-to-head 统一用 `step MiB`，JSON 里同时有 `peak_memory_mib`（step）与 `total_peak_mib`。

### 4.5 时间去哪了（`bench/opt_probe.py` / `bench/vocab_probe.py`）

160.46M / seq 1024 / batch 4、bf16、SDPA，逐段计时（10 次，CUDA 同步）。两列出自
`bench/vocab_probe.py` 的同一次运行（同模型、同输入，只是 loss 实现不同），整步一行仍引自 §4.4 的 profiler：

| 段 | ms（参考 CE） | ms（融合） | 说明 |
|---|---:|---:|---|
| 2 层 trunk（embedding + 2×block + final LN） | 4.12 | 4.12 | attention/MLP 本体 |
| head（1024→65536）前向 | 8.05 | 8.05 | cuBLAS bf16 |
| CE 前向 | 5.69 | — | 参考实现要 fp32 上采样 + `log_softmax` |
| **head+CE 前向** | **13.74** | **8.71** | 1.58× |
| **head+CE 前向+反向** | **38.85** | **25.22** | **1.54×** |
| 其中 CE 段峰值 | 10,964 MiB | 9,428 MiB | 进程内高水位（同进程同模型） |
| 参考 profiler 的整步（compile） | 48.39 | — | §4.4 同窗口 |

结论：

1. **vocab 65536 的投影 + CE 仍是单步最大单项**，但它现在是 **8.71 / 12.83 = 68% 的前向**，
   而参考 CE 是 13.74 / 17.86 = 77%。GEMM 本身（8.05 ms 前向、反向约 2× 前向）是算术决定的，
   融合只能把 CE 那部分（参考 5.69 ms 前向 / 13.74 ms 前向+反向）压到几乎免费 ——
   融合版的 head+CE 前向+反向 25.22 ms 只比 head 单独的 25.11 ms 多 0.1 ms，
   因为 CE 的读写正好落在 matmul 输出写和梯度写这两次 HBM 访问上。
2. **logits 尺寸的显存**：参考 CE 段峰值 10,964 MiB vs 融合 9,428 MiB（−1.5 GiB，含模型常驻）。
   在整步层面（§4.6）同结构是 −2,048 MiB。
3. optimizer 只占整步 5.7%（2.83 ms），所以循环层面没有多少可挤的了。

### 4.6 词汇投影融合：把 CE 写进 logits 缓冲（TP 源码改动）

`tensorplay/nn/_fused_ce.py` 用 Triton 实现“原地 CE”：前向只读两遍 logits（行 max、行 `sum(exp)`），
只留两个 per-row fp32 向量；反向把 logits 缓冲原地改写成 `(softmax - onehot) * g`。
没有 `log_softmax` 输出、没有 fp32 上采样、没有第二个 full-size 缓冲。`F.linear_cross_entropy`
现在真的融合（`reduction` ∈ mean/sum/none、支持 `ignore_index`；带 class weight / label smoothing 走参考路径）。

160.46M / seq 1024 / batch 4、**bf16 权重**（与 §4 head-to-head 同档），3 reps，安静窗口（`bench/fused_ce.sh`）。
这是**专门的 A/B 轮**：它包含 §4 的 compare 列表里没有的基线（编译 trunk + 参考 CE），
所以表内只看本轮的相对关系；同一批配置在 §4 的同窗口整轮里是 85,124 / 97,791 / 97,761，两轮相差 ≤1.4%。

| 配置 | tok/s | step ms | step 峰值 MiB |
|---|---:|---:|---:|
| 参考 CE + `tp.compile` + fused AdamW（原生产配置） | 84,542 | 48.45 | 3,070 |
| 参考 CE，eager + fused AdamW | 78,264 | 52.33 | 4,280 |
| **融合 CE，eager + fused AdamW** | **96,935** | 42.25 | **2,232** |
| 参考 CE，编译 trunk + 编译区外 eager loss + fused AdamW | 77,464 | 52.88 | 4,707 |
| **融合 CE + 编译 trunk + fused AdamW** | **97,292** | 42.10 | **2,659** |

1. **同结构 A/B**：78,264 → 96,935 = **1.239×**，step 快 10.1 ms，step 峰值 **−2,048 MiB**，loss 一致。
2. **编译 trunk 后的同结构 A/B**：77,464 → 97,292 = **1.256×**，峰值 4,707 → 2,659 MiB（同样 **−2,048 MiB**）。
   两种布局省下的显存完全相同，说明省的是 logits 派生缓冲，不是布局差异。
3. **对原生产配置**：84,542 → 97,292 = **1.151×**，峰值 3,070 → 2,659 MiB。
4. **`tp.compile` 的收益几乎全在 CE 上**：只编译 trunk 是 77,464（略低于不编译的 78,264），含 CE 的整步编译才到 84,542，
   与 §4.4 的“+9.5%”自洽。融合之后 compile 落进噪声（97,292 vs 96,935，区间重叠）却多占 427 MiB ——
   **融合 CE 之后这个旋钮就可以关掉了**，最简单的配置就是 eager 融合（96,935 tok/s、2,232 MiB）。
5. **唯一不能让融合 CE 进编译区**：`tp.compile` 捕获会重放 trace 出的算子，kernel 的 per-row 向量在调用内分配，
   会被图冻结在捕获值（实测得到 −0.0055 这种不可能的值）。所以捕获区内自动退回组合参考路径。
   若仍要开 `tp.compile`，就必须用“编译 trunk + 编译区外 eager 应用 loss”的布局（与 Megatron 一致，
   它的 CE 也是 eager）；按第 4 点，这个布局现在并不划算。
6. 数值：fp32 下 loss/梯度 relative 1.5e-7 / 3e-8；bf16 下 loss 逐位一致、梯度 ≤1 ulp。**没有 bf16 CE**。
7. **跨栈是同窗口数字**：`bench/compare.py --preset bf16` 把 §4 的全部 bf16 行一次跑完，
   其中融合 CE 97,791 对 MG+TE 66,956 = **1.461×**、对 MG+TE+FusedAdam 70,254 = **1.392×**，不再是 provisional。
   本节表里的 84,542 / 78,264 / 96,935 / 77,464 / 97,292 是专门的 A/B 轮（含“编译 trunk + 参考 CE”这个
   compare.py 没有的基线），与上面整轮相差 ≤1%（85,124 / — / 97,791 / — / 97,761）。
8. **精度档对照**：`--amp`（fp32 权重 + autocast）下参考 72,029 → 融合 88,839 = **1.233×**、峰值 5,414 → 3,382 MiB；
   两档收益一致（1.24–1.26×），说明不依赖精度策略。
9. **只覆盖纯 CE**：class weight、label smoothing、KL/蒸馏项仍然要 logits，走参考路径
   （`sft_loss_with_head` 显式退回 `sft_loss`）；这些分支没有做吞吐 A/B，上面的倍数不含它们。
证据：[`../bench/results/fused_ce.md`](../bench/results/fused_ce.md)。

## 5. 现在的结论

- **有效证据（TE 2.21.0 已安装、cuDNN FusedAttention 已确认）**：同结构（160.46M / 474.07M 参数）、同 bf16、同 token 预算、同 optimizer 语义下，TensorPlay 原生产路径（SDPA + 固定 shape `tp.compile` + fused AdamW）比 Megatron+TE+FusedAdam（厂商最强配）快 **21.2% / 8.1%**，比 Megatron+TE 默认配置快 **27.1% / 13.4%**。fp32 下快 14.4%。160M 的 bf16 行来自**同一个安静窗口**。
- **新增最强项（TP 本地补丁）**：把 vocab 投影的 CE 融合进 logits 缓冲（§4.6）。160M bf16-权重档：参考 CE 78,264 → 融合 CE **96,935**（**1.239×**，step 峰值 **−2.0 GiB**）；编译 trunk 布局下 77,464 → 97,292（**1.256×**，同样 −2,048 MiB）；对原生产配置 84,542 → 96,935 = **1.147×**。474M：26,823 → **27,866**（**1.039×**，峰值 −1.1 GiB）。loss 与 fp32 参考一致（不是 bf16 CE）。**跨栈是同窗口**：160M 融合 CE 97,791 对 MG+TE 66,956 = **1.461×**、对 MG+TE+FusedAdam 70,254 = **1.392×**（`bench/compare.py --preset bf16` 整轮）；474M 仍跨窗口（约 1.18× / 1.13×），标 provisional。
- **必须同时说清楚的对等事实**：TensorPlay 的 **math** 路径对 Megatron+TE 只有 1.02× / 0.98×（160M；474M 是 0.84×），即**同算术下不占优**；优势来自融合 attention kernel + 图执行，不是训练循环。
- **显存**：160M 下原生产路径 3,070 MiB vs TE 2,769（+301 MiB），融合 CE 后 **2,232 MiB 已低于 TE**；474M 下 6,707 vs 5,382（+1.3 GiB）。但**等有效 batch 16** 时反而是 TensorPlay 省：3,070 vs Megatron 8,851 MiB（2.9×），因为累积不复制激活。分块 CE 能再换显存但掉吞吐。
- **已证明（自身对照）**：megaTrain AMP bf16 比自己的 legacy 路径快 2.24×、省 1.1 GiB，loss 一致。融合 CE 的对照（参考 vs 融合）不依赖任何跨栈假设：1.239×（160M）、1.039×（474M），显存 −2.0 GiB / −1.1 GiB。
- **装不上的部分（硬原因）**：Megatron-Bridge ≥0.4 要求 Python `>=3.12,<3.13`（本机 3.13 被 pip 拒绝），0.3.1 又要求 `transformers<5` / `transformer-engine<2.13`；NeMo-RL 仅占位包且同样锁 Python 3.12。见 [`../bench/results/nvidia_stack_install.md`](../bench/results/nvidia_stack_install.md)。
- **已证伪**：logsumexp CE、分块 CE（吞吐 −10%~−32%）、`native_ignore` 在本环境都不是生产优化；`tp.compile` 只在 `dynamic=True` 时是负优化，固定 shape 下 `dynamic=False` 反而 +8.8%（已做成 `compile_dynamic` 开关，默认 False）。
- **极致优化矩阵（§4.4，160M 同几何）**：唯一显著正向旋钮是固定 shape 的 `tp.compile`（+8.8%）；`max-autotune` / `fullgraph` 与 `reduce-overhead` 等价；整步 CUDA Graph 只有 +1.6% 且与 compile 互斥；fused AdamW 在单步内是噪声，梯度累积 ×4 再 +4.9%（但有效 batch 变为 4×）。`F.linear_cross_entropy` 在 upstream 1.3 是参考实现（不省显存也不加速），本仓库把它换成融合实现后成为最大单项收益（§4.6）。
- **瓶颈定位**：vocab 65536 的输出投影 + CE 占前向 69%（9.03 / 13.03 ms），反向是前向的 2.2×，optimizer 只占整步 5.7% —— 这就是为什么循环层面的优化收益有限。
- **代价与边界**：融合 CE 只能 eager 应用（捕获区会冻结 kernel 的 per-row 缓冲，故自动退回参考路径），这与 Megatron 的布局一致（它的 CE 也是 eager）；且它是**本地 TP 补丁**，不是 upstream 1.3.0 行为（本机私有笔记 `devdocs/local/` 记录了 commit）。精度没有让步：loss 与 fp32 参考一致，梯度差 ≤1 bf16 ulp。
- **精度纪律**：唯一能再挤出 ~4% 的开关是把 CE 从 fp32 降到 bf16（88,056 tok/s，对基线 +13.9%），但它改变 loss 轨迹（10 步后 11.1875 vs 11.2169），而 Megatron 用 fp32 CE，故不纳入 headline。
- **作废**：任何未装 TE 时得到的"1.33×"类结论（见 4.3 附录）。
