# NVIDIA 栈安装记录（base 环境，Python 3.13.13）

目标：在同一个 base 环境里装好 Megatron-Bridge / NeMo-RL / Megatron-Core，用于同配置基准对比。

## 已装成功

| 组件 | 版本 | 来源 |
| --- | --- | --- |
| `megatron-core` | 0.16.1 | PyPI wheel |
| `transformer_engine` | 2.21.0 | 官方 cu13 wheel |
| `transformer_engine_cu13` | 2.21.0 | 官方 cu13 wheel |
| `transformer_engine_torch` | 2.21.0 | **本机源码编译**（cp313 无官方 wheel），34 个目标文件 |
| `nvidia-cublas` | 13.8.2.12 | 升级，因 TE core 库需要 `cublasLtGroupedMatrixLayoutInit_internal`（13.1.1.3 没有） |

细节与符号证据见 [`te_install.md`](te_install.md)。

## 装不上的，以及硬原因

### Megatron-Bridge

```
$ python -m pip download --no-deps megatron-bridge==0.7.1
ERROR: Ignored the following versions that require a different python version:
  0.4.0 … 0.7.1 Requires-Python >=3.12,<3.13
ERROR: Could not find a version that satisfies the requirement megatron-bridge==0.7.1
```

- **所有 ≥0.4 的版本都要求 Python `>=3.12,<3.13`**，本机是 3.13.13，pip 直接拒绝，不是编译失败。
- 唯一可选的 `0.3.1` 依赖与本机可用栈冲突：
  - `transformers<5.0.0`（本机 5.19.0，且这是许多包共享的基础依赖）
  - `transformer-engine<2.13.0`（本机已装 2.21.0，是 cp313 上唯一能编译成功的版本）
  - `nvidia-resiliency-ext~=0.4.1`（镜像无发行版）
  - `mamba-ssm`、`causal-conv1d`（构建环境缺 torch，此前已实测失败）

  强制 `--no-deps` 装 0.3.1 只会在 import/使用时炸，不能算“装上”。

### NeMo-RL

- PyPI 上只有占位 `nemo-rl==0.0.0`（无代码）。
- 官方安装方式是 NVIDIA 容器或从源码装，源码按本机私有笔记（`devdocs/local/`，不入库）记录的 commit clone 作参考。
- 其运行时依赖（vllm、TE、megatron-core 特定组合）同样要求 Python 3.12，和 Bridge 的限制一致。

## 结论

- **可用的 NVIDIA 训练核心 = Megatron-Core 0.16.1 + Transformer Engine 2.21.0（cuDNN `FusedAttention` 后端）**，这也是 Bridge/NeMo-RL 底层真正跑的 kernel 层。
- Bridge 与 NeMo-RL 的编排层在 Python 3.13 装不上；要用它们必须另建 Python 3.12 环境并配 torch 2.9 / TE 2.10 / transformers 4.x，那会是**另一套栈**，不能拿来做同环境 AB。
- 基准因此以 Megatron-Core + TE 为 NVIDIA 侧代表；TE 的融合 attention/LayerNorm 已确认被选中（见 `te_install.md` 第 4 节）。
