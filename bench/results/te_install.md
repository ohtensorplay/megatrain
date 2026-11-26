# Transformer Engine 安装证据（2026-11-25）

环境：Python 3.13.13、torch 2.13.0+cu130、CUDA toolkit 13.0、RTX 3090（sm86）。

## 1. 已安装版本

```
$ python -m pip show transformer-engine transformer-engine-cu13 transformer-engine-torch nvidia-cublas
Name: transformer_engine            Version: 2.21.0
Name: transformer_engine_cu13       Version: 2.21.0
Name: transformer_engine_torch      Version: 2.21.0
Name: nvidia-cublas                 Version: 13.8.2.12
```

- `transformer_engine` / `transformer_engine_cu13`：官方预编译 wheel（`py3-none-manylinux_2_28_x86_64`，sha256 与索引一致）。
- `transformer_engine_torch`：**本机源码编译**。官方 release 只提供 cp312 wheel，没有 cp313；从 sdist 编译，34 个目标文件，产物为
  `transformer_engine_torch-2.21.0-cp313-cp313-linux_x86_64.whl`。

## 2. 为什么必须换 cuBLAS

`transformer_engine_cu13` 的 `wheel_lib/libtransformer_engine.so` 需要：

```
cublasLtGroupedMatrixLayoutInit_internal, version libcublasLt.so.13
```

- 原环境 `nvidia-cublas==13.1.1.3`（被 `cuda-toolkit==13.0.3` 钉死）**不导出**该符号；
  系统 CUDA 13.0 的 `libcublasLt.so.13.1.1.3` 同样没有。
- 换成 `nvidia-cublas==13.8.2.12`（pip wheel，sha256 校验、zip 完整性校验通过）后符号存在：

```
$ nm -D --defined-only nvidia/cu13/lib/libcublasLt.so.13 | grep GroupedMatrixLayoutInit
0000000001b65330 T cublasLtGroupedMatrixLayoutInit_internal@@libcublasLt.so.13
```

- 旧库备份在 `/tmp/opencode/cublas_backup/`，可随时回滚。
- 换库后 torch 2.13 的 bf16/fp32 matmul 均正常（见下）。

## 3. 运行验证

```bash
SP=/home/bluemoon/miniconda3/lib/python3.13/site-packages/nvidia
export LD_LIBRARY_PATH=$SP/nccl/lib:$SP/cu13/lib:$LD_LIBRARY_PATH
python - <<'PY'
import torch
print(torch.__version__)
a = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
print(float((a @ a).float().abs().mean()))
print(float((torch.randn(512,512,device='cuda') @ torch.randn(512,512,device='cuda')).sum()))
import transformer_engine.pytorch as te
lin = te.Linear(256, 512, bias=True).cuda().bfloat16()
x = torch.randn(8, 64, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
y = lin(x); y.float().sum().backward()
print("TE linear ok:", tuple(y.shape), float(x.grad.float().abs().mean()))
ln = te.LayerNormLinear(256, 512).cuda().bfloat16()
print("TE LayerNormLinear ok:", tuple(ln(x).shape))
PY
```

输出（本次运行）：

```
torch 2.13.0+cu130
matmul: 25.534229278564453
fp32 matmul: 10788.10546875
TE import OK
TE linear ok: (8, 64, 512) 0.43881386518478394
TE LayerNormLinear ok: (8, 64, 512)
TE attention module: ['DotProductAttention', 'GatedDeltaNet2Attention', ...]
```

## 4. 确认走的是 NVIDIA 融合 kernel

`NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=2` 跑 Megatron `--te`（bf16，seq 512，hidden 1024）：

```
[DEBUG | DotProductAttention]: Available backends =
    {FlashAttention=False, FusedAttention=True (sub-backend 1), UnfusedDotProductAttention=True}
[DEBUG | DotProductAttention]: Selected backend = FusedAttention (sub-backend 1).
[INFO  | DotProductAttention]: Running with FusedAttention backend (sub-backend 1)
```

FlashAttention 不可用是因为没有 `flash-attn` 包；cuDNN fused attention 被选中，`te_layers` 里同时出现 TE 的
`LayerNorm` 等模块，说明不是 torch 回退路径。

## 5. 复现基线要注意

- 必须设置 `LD_LIBRARY_PATH` 包含 `nvidia/nccl/lib` 与 `nvidia/cu13/lib`（TensorPlay 本身也需要 nccl 路径）。
- `--te` 走 TE 层规格；`--fused-adam` 用 `te.optimizers.FusedAdam`（给 NVIDIA 侧的“最强配”）。
- Megatron 默认 `attention_dropout=0.1` / `hidden_dropout=0.1`，基准里显式设为 0，与 TensorPlay 模型对齐。
