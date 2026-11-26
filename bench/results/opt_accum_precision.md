# 有效 batch 16 与精度口径探针（RTX 3090，160.46M，同模型）

两组补充实验，用来说明"极致优化"里哪些收益是真实的、哪些只是换了精度口径。

```bash
bash bench/opt_extra.sh
```

守卫同 `opt_sweep.sh`：每次计时前要求 GPU `util=0` 且 `free≥18 GiB`。以下是脚本的原始输出（3 次/配置，`--seed 0` 固定初始化）。

## 1. 有效 batch 16：累积 vs 直接大 batch

TensorPlay：micro-batch 4 × accum 4 = 有效 batch 16，仍走 compile。
Megatron：直接把 micro-batch 开到 16（它内部自己处理累积）。

| 栈 | 配置 | tokens/s（3 次） | step ms | step 峰值 MiB |
|---|---|---:|---:|---:|
| TensorPlay | bf16 SDPA + compile + accum 4（有效 batch 16） | 88,280 / 88,205 / 88,251 | 185.65 | 3,070 |
| Megatron-Core | bf16 + TE + FusedAdam（batch 16） | 78,611 / 78,584 / 78,567 | 208.49 | 8,851 |

- 中位数：**88,251 vs 78,584 = 1.123×**。
- **显存差距在等有效 batch 下更大**：TensorPlay 3,070 MiB vs Megatron 8,851 MiB（−65%）。累积只多跑 4 次前反向，不复制激活；Megatron 的 batch 16 一次性持有 4× 激活。
- 所以"batch 4 下 1.22×"不是唯一结论：有效 batch 放大后吞吐比回落到 1.12×，但显存优势变成 2.9×。同一实验的两种读法，不矛盾。

## 2. 精度口径：`--no-autocast` 到底快在哪

同一 seed（`--seed 0`）、10 步、batch 4、compile。

| 配置 | tokens/s（3 次） | step ms | step 峰值 MiB | 10 步后 loss |
|---|---:|---:|---:|---:|
| autocast bf16（LN/CE 走 fp32，默认） | 84,545 / 83,327 / 84,061 | 48.73 | 3,070 | 11.21689 |
| 无 autocast + `--ce-fp32`（LN bf16，CE fp32） | 83,982 / 84,374 / 84,301 | 48.59 | 3,070 | 11.21693 |
| 无 autocast（LN bf16，CE bf16） | 89,028 / 89,444 / 89,192 | 45.92 | 2,558 | **11.18750** |

结论：

1. 去掉 autocast 包装本身**不省时间**（84,301 ≈ 84,061），省下的 dispatch 被显式 `logits.float()` 抵消。
2. 真正的 +6% 来自**把 CE 的 log_softmax 从 fp32 降到 bf16**（89,192 tok/s），同时 step 峰值 −512 MiB。但它改变了数值：10 步后 loss 11.1875 vs 11.2169，是精度差不是噪声。
3. Megatron 的 `compute_language_model_loss` 会把 logits **上采样到 fp32** 再算 CE；所以拿 bf16-CE 的 89k 去对 Megatron 的 fp32-CE 基线不是同精度比较。
4. 因此 **headline 仍用 autocast/fp32-CE 口径（84,247）**；`--no-autocast` 只作为"精度换速度"的可选旋钮记录，不进默认路径。

## 3. 原始输出

```text
# 1. equal effective batch 16
### TP accum 4 x batch 4 + compile
  88,280	 185.59	   3070	loss=11.22915
  88,205	 185.75	   3070	loss=11.22917
  88,251	 185.65	   3070	loss=11.22912
### MG batch 16
  78,611	 208.42	   8851	loss=11.1964
  78,584	 208.49	   8851	loss=11.20209
  78,567	 208.53	   8851	loss=11.20468
# 2. precision accounting (batch 4, 10 steps, seed 0)
### TP compile, autocast (fp32 LN/CE)
  84,545	  48.45	   3070	loss=11.21689
  83,327	  49.16	   3070	loss=11.21688
  84,061	  48.73	   3070	loss=11.21691
### TP compile, no-autocast + fp32 CE
  83,982	  48.77	   3070	loss=11.21693
  84,374	  48.55	   3070	loss=11.21693
  84,301	  48.59	   3070	loss=11.21692
### TP compile, no-autocast (bf16 LN/CE)
  89,028	  46.01	   2558	loss=11.1875
  89,444	  45.79	   2558	loss=11.1875
  89,192	  45.92	   2558	loss=11.1875
```
