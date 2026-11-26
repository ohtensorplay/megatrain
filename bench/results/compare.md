# ⚠️ 无效证据：TE 未安装时的 torch 回退路径

> 这份输出不能用于“击败 NVIDIA”的结论；有效数据见 `compare_te.md`。

# 原始输出（12 step × 3 次，RTX 3090）

命令：`python bench/compare.py --steps 12 --repeat 3`

transformer_engine available: False
bf16 + unfused math attn                    58,507 tok/s    70.01 ms     3,361 MiB  ±0.8%
bf16 + math attn (identical arithmetic)     68,234 tok/s    60.03 ms     4,647 MiB  ±0.8%
bf16 + SDPA attn (production)               77,662 tok/s    52.74 ms     4,280 MiB  ±0.2%
fp32 params + bf16 AMP + SDPA               71,932 tok/s    56.94 ms     5,414 MiB  ±0.6%
fp32                                        27,038 tok/s   151.49 ms     4,417 MiB  ±0.3%
fp32 + math attn                            30,950 tok/s   132.34 ms     5,710 MiB  ±0.2%

| stack | setup | params M | tokens/s | step ms | peak MiB | spread |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| megatron-core | bf16 + unfused math attn | 160.46 | 58,507 | 70.01 | 3,361 | ±0.8% |
| tensorplay | bf16 + math attn (identical arithmetic) | 160.46 | 68,234 | 60.03 | 4,647 | ±0.8% |
| tensorplay | bf16 + SDPA attn (production) | 160.46 | 77,662 | 52.74 | 4,280 | ±0.2% |
| tensorplay | fp32 params + bf16 AMP + SDPA | 160.46 | 71,932 | 56.94 | 5,414 | ±0.6% |
| megatron-core | fp32 | 160.46 | 27,038 | 151.49 | 4,417 | ±0.3% |
| tensorplay | fp32 + math attn | 160.46 | 30,950 | 132.34 | 5,710 | ±0.2% |

| comparison (same dtype, same architecture, same token budget) | speedup |
| --- | ---: |
| TensorPlay `bf16 + math attn (identical arithmetic)` vs Megatron `bf16 + unfused math attn` | 1.166× |
| TensorPlay `fp32 + math attn` vs Megatron `fp32` | 1.145× |
| TensorPlay `bf16 + SDPA attn (production)` vs Megatron `bf16 + unfused math attn` | 1.327× |

wrote bench/results/compare.json
