# 词汇投影融合 CE：TP 源码改动 + 实测

日期：2026-11-26。目标：“vocab 投影优化”——160M 几何下 `head + CE` 占前向 69%、单步 9.03 ms，
是单步最大单项，也是 CE workspace 显存的主因。

## 1. 改了什么（TensorPlay 源码）

新文件 `tensorplay/nn/_fused_ce.py`（约 240 行，Triton kernel + autograd Function）：

- 前向只读 logits**两遍**（行 max，然后 `sum(exp(x-max))`），只留下两个 per-row fp32 向量
  （`max`、`sum`），**不写任何 full-size 缓冲**；
- 反向把 logits 缓冲**原地**改写成梯度：`dlogits = (exp(x-max)/sum - onehot) * g`；
- loss 全程 fp32：`log(sum(exp(x-max))) + max - x[t]`，与 `log_softmax + nll_loss` 同式同累积精度。

`F.linear_cross_entropy` 现在真的融合：先 `F.linear` 出 logits（唯一的 full-size 缓冲），
再在同一个缓冲上算 CE 与反向。`reduction` 支持 mean/sum/none，支持 `ignore_index`；
`weight` / `label_smoothing != 0` / 非 2-D / 非 CUDA 走原来的组合参考路径。

关键陷阱（已修）：`ctx` 持有 logits 张量会与 node 形成引用环，每步泄漏一个 512 MiB 缓冲
（10 步 +4.7 GiB，且 step 峰值虚高到 10.1 GiB）。反向用完即 `ctx.* = None`，稳态回到平线。

作用域限制（已写进 docstring）：这是**eager 路径**优化。`tp.compile` 的捕获会重放 trace 出的算子，
而本 kernel 的 per-row 向量在调用内分配，图里会被冻结在捕获值 → 在捕获区里自动退回
`cross_entropy(linear(...))`（算术一致、可被完整 trace）。因此生产配置是
**编译 trunk + 在编译区外 eager 应用 loss**（与 Megatron 的布局一致），§2 给出该结构下的 A/B。

另一条语义边界：反向**消耗** logits 缓冲，所以同一张图只能走一次。`retain_graph=True` 后再
backward 一次、或对同一张图第二次 `autograd.grad`，都会在“梯度”上再求导并静默给出错误结果；
要留下 logits 必须先 `clone()`。这条已写进 `fused_cross_entropy` 的 docstring。

## 2. 160.46M / seq 1024 / batch 4（**bf16 权重**，3 reps，2026-11-26）

同结构、同 dtype、同 token 预算；每 rep 前检查 GPU `util=0`，扰动窗口丢弃。
（`--precision bf16`，无 `--amp`，与 §4 head-to-head 的 bf16 行同档。）

| 配置 | tok/s（3 reps） | 中位 | step ms | step 峰值 MiB |
|---|---:|---:|---:|---:|
| 参考 CE + `tp.compile` + fused AdamW（原生产配置） | 84,542 / 84,636 / 84,194 | **84,542** | 48.45 | 3,070 |
| 参考 CE，eager + fused AdamW（同结构基线） | 78,264 / 78,232 / 78,278 | **78,264** | 52.33 | 4,280 |
| **融合 CE，eager + fused AdamW** | 98,739 / 95,447 / 96,935 | **96,935** | 42.25 | **2,232** |
| 参考 CE，编译 trunk + 编译区外 eager loss + fused AdamW | 77,506 / 77,464 / 77,360 | **77,464** | 52.88 | 4,707 |
| **融合 CE + 编译 trunk + fused AdamW** | 98,369 / 96,986 / 97,292 | **97,292** | 42.10 | **2,659** |
| 融合 CE，eager（`compare.py --preset bf16` 整轮，同窗口） | 98,057 / 97,791 / 97,170 | **97,791** | 41.89 | **2,232** |

1. **同结构 A/B（唯一干净的比较）**：参考 78,264 → 融合 96,935 = **1.239×**，step 52.33 → 42.25 ms（−10.1 ms），
   **step 峰值 −2,048 MiB（−47.9%）**，loss 一致（11.21086 vs 11.21085）。
2. **编译 trunk 后的同结构 A/B**：77,464 → 97,292 = **1.256×**，峰值 4,707 → 2,659 MiB（**−2,048 MiB**）。
   两种布局下融合 CE 省的显存完全相同（−2.0 GiB），说明它省的是 logits 派生缓冲，不是别的东西。
3. **对原生产配置**（参考 CE + 整步 `tp.compile`）：84,542 → 97,292 = **1.151×**，step 48.45 → 42.10 ms，
   峰值 3,070 → 2,659 MiB（**−411 MiB**）。
4. **`tp.compile` 的收益几乎全在 CE 上**：只编译 trunk 是 77,464，反而略低于完全不编译的 78,264；
   含 CE 的整步编译才到 84,542。融合之后 compile 落进噪声（97,292 vs 96,935，两组区间重叠），
   而且编译 trunk 要多占 427 MiB —— **融合 CE 之后它不再值得开**。
5. **跨栈（同窗口）**：`bench/compare.py --preset bf16` 把 Megatron 与 TensorPlay 的全部 bf16 行放在
   **同一个安静窗口**里跑完：融合 CE 97,791 对 MG+TE 66,956 = **1.461×**、对 MG+TE+FusedAdam 70,254 = **1.392×**
   （编译 trunk 版 97,761 = 1.460× / 1.392×）。这取代了此前跨窗口的 provisional 数字（那版是 1.477× / 1.410×）。
6. **精度档对照（`--amp` = fp32 权重 + autocast）**：参考 72,029 → 融合 88,839 = **1.233×**、峰值 5,414 → 3,382 MiB。
   两档的融合收益一致（1.24–1.26×），说明它不依赖精度策略。
7. **复测（同配置、同窗口，3 rep）**：96,935 那一行重跑是 **97,691 / 97,466 / 97,868** tok/s，
   step 41.85–42.02 ms、峰值仍 2,232 MiB、loss 11.21082–11.21087（与首次跑逐位同区间）。
   两次合计 6 rep 的分布是 95.4k–98.7k，所以正文统一用**偏保守的首次中位 96,935** 报告；
   若用复测中位则是 1.248×（同结构），结论不变。
8. **bench 走的是库里的同一条路**：`--fused-ce` 调 `F.linear_cross_entropy`，
   `sft_loss_with_head` 也调它（差别只是后者用 `reduction="none"` + mask 归一化 + shift），没有 bench 专用实现。
   bench 的合成任务是**未 shift**的逐 token CE（各栈同此约定，所以横向比不受影响）；库目标的 shift 与 mask
   语义由 `tests/test_fused_ce.py` 的等效性测试覆盖（loss/梯度与 `sft_loss` 一致）。

## 3. 474.07M / seq 2048 / batch 2（bf16 权重，2026-11-26）

| 配置 | tok/s（reps） | step ms | step 峰值 MiB | loss |
|---|---:|---:|---:|---:|
| 参考 CE + `tp.compile` + fused AdamW | 26,823 / 26,830 / 26,812 | 152.71 | 6,703 | 11.10488 |
| 参考 CE，eager + fused AdamW | 26,674 / 26,659 | 153.56 | 6,910 | 11.10480 |
| **融合 CE + 编译 trunk + fused AdamW** | 27,783 / 27,949 | 147.43 | **5,607** | 11.10488 |

- 对原生产配置：26,823 → **27,866**（两次均值）= **1.039×**，step 152.7 → 147.4 ms，峰值 6,703 → 5,607 MiB（**−1,096 MiB**）。
- 规模越大，CE 在单步中的占比越小：160M 是 trunk 4.1 ms 对 head+CE 9.0 ms；474M 的 trunk 大 4 倍而 head+CE 尺寸不变，
  所以同样的 CE 优化在 474M 只值 3.9%。这是预期方向，不是测量差异。
- 这几 rep 恰逢邻居 util 57–100%，但两次 fused rep 只差 0.6%，且 474M 是算力瓶颈（不是 HBM/调度瓶颈），未丢弃。

## 4. 数值验证

- fp32 logits：loss 相对误差 1.5e-7，梯度 3e-8；vocab 3 / 100 / 4097（非 block 对齐）均通过；
- bf16 logits：loss 逐位一致（bit-identical，两个几何各 2 次），梯度差 ≤1 bf16 ulp；
- `ignore_index`：被忽略的行梯度恰为 0，loss 与参考一致（`tests/test_fused_ce.py`）；
- 4096×65536 的 CE 段：13.76 → 4.4 ms（3.1×），CE 段峰值 8,193 → 5,633 MiB。

## 5. 证据边界

- 融合 CE **不能**进入 `tp.compile` 捕获区（见 §1 作用域限制）；本表的 fused 行都是 eager loss。
- 这是**本地 TP 源码改动**，不是 upstream 1.3.0 release 的行为；已安装的 1.3.0 release 不含它。
  README/文档里凡引用该数字处都注明 “TP + local patch”。
- 只融合了**纯 CE**路径：class weight、`label_smoothing != 0`、以及 `sft_loss` 的 KL/蒸馏项需要 logits 本身，
  仍走参考实现（`sft_loss_with_head` 会显式退回 `sft_loss`，有测试覆盖）。
  这些分支**没有做吞吐 A/B**，所以上面任何倍数都不包含它们。
- 与 Megatron 的比较仍只在 README 的现有表格里；本文件只量化“融合 vs 不融合”，两者都是 TensorPlay。

## 6. 复现

```bash
# 本机临时目录由环境给出，不写死路径
TMPDIR=${TMPDIR:-$(mktemp -d)}

# §2 / §3 的 5 个配置（AMP=0 = bf16 权重；AMP=1 = fp32 权重 + bf16 autocast）
AMP=0 GEOM=160m REPS=3 STEPS=12 MIN_FREE=8000 OUT=$TMPDIR/fused_160m.log ./bench/fused_ce.sh
AMP=0 GEOM=474m REPS=3 STEPS=8  MIN_FREE=17000 OUT=$TMPDIR/fused_474m.log ./bench/fused_ce.sh
AMP=1 GEOM=160m REPS=3 STEPS=12 MIN_FREE=8000 OUT=$TMPDIR/fused_160m_amp.log ./bench/fused_ce.sh

# §4 的逐段计时与 kernel 微基准
PYTHONPATH=bench python bench/vocab_probe.py --iters 10

# 跨栈同窗口（Megatron + TE / + FusedAdam 与 TensorPlay 各配置交替跑，
# 每 rep 前要求 util=0、空闲显存达标，被邻居扰动的窗口丢弃重测）
MG=1 AMP=0 GEOM=160m REPS=3 STEPS=12 MIN_FREE=8000 ./bench/fused_ce.sh
```

`bench/mg_vs_fused_retry.sh` 把上面跨栈那条包成 8 次重试（要求凑满 21 个干净 rep），
适合在共享 GPU 上排队。
