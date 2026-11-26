# ⚠️ 无效证据：TE 未安装时的 torch 回退路径（474M）

> 有效数据见 `compare_te_large.md`。

# 大模型几何：474.07M 参数，seq 2048（RTX 3090，8 step）

```bash
python bench/compare.py --steps 8 --repeat 3 --preset bf16 --seq 2048 --batch 2 \
    --hidden 2048 --layers 4 --heads 16 --vocab 65536
```

Megatron 3 次：16830.7 / 16934.0 / 16912.4 tok/s
TensorPlay math 4 次：19711.9 / 19811.8 / 19911.8 / 19780.0 tok/s
TensorPlay SDPA 4 次：26952.0 / 26750.9 / 26618.0 / 26590.5 tok/s

原始 JSON 行：
```
  "params_m": 474.07,
  "step_ms": 207.79,
  "tokens_per_second": 19711.9,
  "peak_memory_mib": 9917.0
  "params_m": 474.07,
  "step_ms": 151.97,
  "tokens_per_second": 26952.0,
  "peak_memory_mib": 6910.0
  "params_m": 474.07,
  "step_ms": 240.31,
  "tokens_per_second": 17044.6,
  "peak_memory_mib": 9545.6
  "params_m": 474.07,
  "step_ms": 205.71,
  "tokens_per_second": 19911.8,
  "peak_memory_mib": 9917.0
  "params_m": 474.07,
  "step_ms": 153.12,
  "tokens_per_second": 26750.9,
  "peak_memory_mib": 6910.0
```
