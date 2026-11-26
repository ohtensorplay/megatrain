# 160.46M 参数、seq 1024，**已装 Transformer Engine**（RTX 3090，12 step × 3 次中位数）

## 2026-11-26 同窗口复测（bf16 权重，含融合 CE）

```bash
python bench/compare.py --steps 12 --repeat 3 --preset bf16 --min-free-mib 20000 \
    --json-out bench/results/compare_te.json
```

这一轮把**所有 bf16 行放在同一个安静窗口**里跑完（守卫：每 rep 前 `util=0` 且空闲 ≥20 GiB，扰动窗口丢弃重测），
所以融合 CE 与 Megatron 的倍数是**同窗口**数字，不再需要“provisional/跨窗口”的注解。
fp32 / AMP 两档不在此轮（`--preset bf16` 过滤掉了），其数字见文末附录（上一轮全旋钮运行）。

```
transformer_engine available: True; geometry {'seq': 1024, 'batch': 4, 'hidden': 1024, 'layers': 2, 'heads': 8, 'vocab': 65536}
  run 1/3: 67,016 tok/s
  run 2/3: 66,956 tok/s
  run 3/3: 66,586 tok/s
bf16 + Transformer Engine                   66,956 tok/s    61.17 ms     2,769 MiB  ±0.6%
  run 1/3: 70,075 tok/s
  run 2/3: 70,254 tok/s
  run 3/3: 70,270 tok/s
bf16 + TE + FusedAdam (vendor best)         70,254 tok/s    58.30 ms     3,377 MiB  ±0.3%
  run 1/3: 59,557 tok/s
  run 2/3: 59,862 tok/s
  run 3/3: 59,706 tok/s
bf16 + unfused math attn                    59,706 tok/s    68.60 ms     3,261 MiB  ±0.5%
  run 1/3: 68,589 tok/s
GPU busy (23847 MiB free, util 100%); waiting…
GPU is quiet again.
  run 2/3: 68,543 tok/s
GPU busy (23847 MiB free, util 48%); waiting…
GPU is quiet again.
  run 3/3: 68,632 tok/s
bf16 + math attn (identical arithmetic)     68,589 tok/s    59.72 ms     4,647 MiB  ±0.1%
GPU busy (23847 MiB free, util 100%); waiting…
GPU is quiet again.
  run 1/3: 77,410 tok/s
GPU busy (23847 MiB free, util 100%); waiting…
GPU is quiet again.
  run 2/3: 77,932 tok/s
  run 3/3: 77,490 tok/s
bf16 + SDPA attn (production)               77,490 tok/s    52.86 ms     4,280 MiB  ±0.7%
GPU busy (23847 MiB free, util 92%); waiting…
GPU is quiet again.
  run 1/3: 85,102 tok/s
  run 2/3: 84,288 tok/s
GPU busy (23847 MiB free, util 52%); waiting…
GPU is quiet again.
  run 3/3: 84,248 tok/s
bf16 + SDPA + tp.compile(dynamic=False)     84,288 tok/s    48.60 ms     3,070 MiB  ±1.0%
  run 1/3: 85,124 tok/s
  run 2/3: 85,124 tok/s
GPU busy (23847 MiB free, util 80%); waiting…
GPU is quiet again.
  run 3/3: 84,384 tok/s
bf16 + SDPA + compile + fused AdamW         85,124 tok/s    48.12 ms     3,070 MiB  ±0.9%
  run 1/3: 98,057 tok/s
  run 2/3: 97,791 tok/s
  run 3/3: 97,170 tok/s
bf16 + SDPA + fused CE + fused AdamW (best)    97,791 tok/s    41.89 ms     2,232 MiB  ±0.9%
GPU busy (23847 MiB free, util 100%); waiting…
GPU is quiet again.
  run 1/3: 98,352 tok/s
  run 2/3: 97,761 tok/s
  run 3/3: 97,354 tok/s
bf16 + SDPA + compiled trunk + fused CE + fused AdamW    97,761 tok/s    41.90 ms     2,659 MiB  ±1.0%

| stack | setup | params M | tokens/s | step ms | step peak MiB | total peak MiB | spread |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| megatron-core | bf16 + Transformer Engine | 160.46 | 66,956 | 61.17 | 2,769 | 2,769 | ±0.6% |
| megatron-core | bf16 + TE + FusedAdam (vendor best) | 160.46 | 70,254 | 58.30 | 3,377 | 3,377 | ±0.3% |
| megatron-core | bf16 + unfused math attn | 160.46 | 59,706 | 68.60 | 3,261 | 3,261 | ±0.5% |
| tensorplay | bf16 + math attn (identical arithmetic) | 160.46 | 68,589 | 59.72 | 4,647 | 4,647 | ±0.1% |
| tensorplay | bf16 + SDPA attn (production) | 160.46 | 77,490 | 52.86 | 4,280 | 4,280 | ±0.7% |
| tensorplay | bf16 + SDPA + tp.compile(dynamic=False) | 160.46 | 84,288 | 48.60 | 3,070 | 4,606 | ±1.0% |
| tensorplay | bf16 + SDPA + compile + fused AdamW | 160.46 | 85,124 | 48.12 | 3,070 | 4,606 | ±0.9% |
| tensorplay | bf16 + SDPA + fused CE + fused AdamW (best) | 160.46 | 97,791 | 41.89 | 2,232 | 2,232 | ±0.9% |
| tensorplay | bf16 + SDPA + compiled trunk + fused CE + fused AdamW | 160.46 | 97,761 | 41.90 | 2,659 | 2,651 | ±1.0% |

| TensorPlay (bf16) vs Megatron (bf16) | tokens/s | vs vendor |
| --- | ---: | ---: |
| `bf16 + math attn (identical arithmetic)` vs `bf16 + Transformer Engine` | 68,589 vs 66,956 | 1.024× |
| `bf16 + math attn (identical arithmetic)` vs `bf16 + TE + FusedAdam (vendor best)` | 68,589 vs 70,254 | 0.976× |
| `bf16 + math attn (identical arithmetic)` vs `bf16 + unfused math attn` | 68,589 vs 59,706 | 1.149× |
| `bf16 + SDPA attn (production)` vs `bf16 + Transformer Engine` | 77,490 vs 66,956 | 1.157× |
| `bf16 + SDPA attn (production)` vs `bf16 + TE + FusedAdam (vendor best)` | 77,490 vs 70,254 | 1.103× |
| `bf16 + SDPA attn (production)` vs `bf16 + unfused math attn` | 77,490 vs 59,706 | 1.298× |
| `bf16 + SDPA + tp.compile(dynamic=False)` vs `bf16 + Transformer Engine` | 84,288 vs 66,956 | 1.259× |
| `bf16 + SDPA + tp.compile(dynamic=False)` vs `bf16 + TE + FusedAdam (vendor best)` | 84,288 vs 70,254 | 1.200× |
| `bf16 + SDPA + tp.compile(dynamic=False)` vs `bf16 + unfused math attn` | 84,288 vs 59,706 | 1.412× |
| `bf16 + SDPA + compile + fused AdamW` vs `bf16 + Transformer Engine` | 85,124 vs 66,956 | 1.271× |
| `bf16 + SDPA + compile + fused AdamW` vs `bf16 + TE + FusedAdam (vendor best)` | 85,124 vs 70,254 | 1.212× |
| `bf16 + SDPA + compile + fused AdamW` vs `bf16 + unfused math attn` | 85,124 vs 59,706 | 1.426× |
| `bf16 + SDPA + fused CE + fused AdamW (best)` vs `bf16 + Transformer Engine` | 97,791 vs 66,956 | 1.461× |
| `bf16 + SDPA + fused CE + fused AdamW (best)` vs `bf16 + TE + FusedAdam (vendor best)` | 97,791 vs 70,254 | 1.392× |
| `bf16 + SDPA + fused CE + fused AdamW (best)` vs `bf16 + unfused math attn` | 97,791 vs 59,706 | 1.638× |
| `bf16 + SDPA + compiled trunk + fused CE + fused AdamW` vs `bf16 + Transformer Engine` | 97,761 vs 66,956 | 1.460× |
| `bf16 + SDPA + compiled trunk + fused CE + fused AdamW` vs `bf16 + TE + FusedAdam (vendor best)` | 97,761 vs 70,254 | 1.392× |
| `bf16 + SDPA + compiled trunk + fused CE + fused AdamW` vs `bf16 + unfused math attn` | 97,761 vs 59,706 | 1.637× |

wrote bench/results/compare_te.json
```

## 附录：上一轮全旋钮运行（含 fp32 与 fp32+AMP，较早窗口）

```bash
python bench/compare.py --steps 12 --repeat 3
```

与上面的 bf16 窗口相比，同名 bf16 行相差 ≤2%（同一个共享 GPU，不同安静窗口）。
`(best)` 行当时指“SDPA + 整步 compile + fused AdamW”；融合 CE 加入后，`(best)` 已被新窗口里的
“fused CE + fused AdamW”取代（旧行仍保留，便于核对旧结论）。

transformer_engine available: True; geometry {'seq': 1024, 'batch': 4, 'hidden': 1024, 'layers': 2, 'heads': 8, 'vocab': 65536}
GPU busy (7025 MiB free, util 0%); waiting…
GPU is quiet again.
  run 1/3: 65,753 tok/s
  run 2/3: 65,871 tok/s
  run 3/3: 66,219 tok/s
bf16 + Transformer Engine                   65,871 tok/s    62.18 ms     2,769 MiB  ±0.7%
  run 1/3: 68,976 tok/s
  run 2/3: 68,912 tok/s
  run 3/3: 69,485 tok/s
bf16 + TE + FusedAdam (vendor best)         68,976 tok/s    59.38 ms     3,377 MiB  ±0.8%
  run 1/3: 59,349 tok/s
  run 2/3: 61,291 tok/s
  run 3/3: 59,538 tok/s
bf16 + unfused math attn                    59,538 tok/s    68.80 ms     3,261 MiB  ±3.3%
  run 1/3: 67,931 tok/s
GPU busy (23847 MiB free, util 100%); waiting…
GPU is quiet again.
  run 2/3: 67,288 tok/s
  run 3/3: 67,873 tok/s
bf16 + math attn (identical arithmetic)     67,873 tok/s    60.35 ms     4,647 MiB  ±0.9%
GPU busy (23847 MiB free, util 99%); waiting…
GPU is quiet again.
  run 1/3: 77,005 tok/s
  run 2/3: 77,487 tok/s
GPU busy (23847 MiB free, util 100%); waiting…
GPU is quiet again.
  run 3/3: 76,923 tok/s
bf16 + SDPA attn (production)               77,005 tok/s    53.19 ms     4,280 MiB  ±0.7%
GPU busy (23847 MiB free, util 100%); waiting…
GPU is quiet again.
  run 1/3: 83,452 tok/s
  run 2/3: 83,883 tok/s
GPU busy (23847 MiB free, util 43%); waiting…
GPU is quiet again.
  OOM (GPU got busy mid-run); retrying after a quiet window…
  run 3/3: 84,494 tok/s
bf16 + SDPA + tp.compile(dynamic=False)     83,883 tok/s    48.83 ms     3,070 MiB  ±1.2%
  run 1/3: 84,203 tok/s
  run 2/3: 84,537 tok/s
GPU busy (23847 MiB free, util 44%); waiting…
GPU is quiet again.
  run 3/3: 84,247 tok/s
bf16 + SDPA + compile + fused AdamW (best)    84,247 tok/s    48.62 ms     3,070 MiB  ±0.4%
  run 1/3: 70,990 tok/s
GPU busy (23847 MiB free, util 99%); waiting…
GPU is quiet again.
  run 2/3: 71,171 tok/s
  run 3/3: 71,596 tok/s
fp32 params + bf16 AMP + SDPA               71,171 tok/s    57.55 ms     5,414 MiB  ±0.9%
  run 1/3: 27,207 tok/s
  run 2/3: 27,368 tok/s
  run 3/3: 27,294 tok/s
fp32                                        27,294 tok/s   150.07 ms     4,077 MiB  ±0.6%
  run 1/3: 31,244 tok/s
GPU busy (23847 MiB free, util 100%); waiting…
GPU is quiet again.
  run 2/3: 31,211 tok/s
  run 3/3: 31,197 tok/s
fp32 + math attn                            31,211 tok/s   131.24 ms     5,710 MiB  ±0.2%
| stack | setup | params M | tokens/s | step ms | step peak MiB | total peak MiB | spread |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| megatron-core | bf16 + Transformer Engine | 160.46 | 65,871 | 62.18 | 2,769 | 2,769 | ±0.7% |
| megatron-core | bf16 + TE + FusedAdam (vendor best) | 160.46 | 68,976 | 59.38 | 3,377 | 3,377 | ±0.8% |
| megatron-core | bf16 + unfused math attn | 160.46 | 59,538 | 68.80 | 3,261 | 3,261 | ±3.3% |
| tensorplay | bf16 + math attn (identical arithmetic) | 160.46 | 67,873 | 60.35 | 4,647 | 4,647 | ±0.9% |
| tensorplay | bf16 + SDPA attn (production) | 160.46 | 77,005 | 53.19 | 4,280 | 4,280 | ±0.7% |
| tensorplay | bf16 + SDPA + tp.compile(dynamic=False) | 160.46 | 83,883 | 48.83 | 3,070 | 4,606 | ±1.2% |
| tensorplay | bf16 + SDPA + compile + fused AdamW (best) | 160.46 | 84,247 | 48.62 | 3,070 | 4,606 | ±0.4% |
| tensorplay | fp32 params + bf16 AMP + SDPA | 160.46 | 71,171 | 57.55 | 5,414 | 5,414 | ±0.9% |
| megatron-core | fp32 | 160.46 | 27,294 | 150.07 | 4,077 | 4,077 | ±0.6% |
| tensorplay | fp32 + math attn | 160.46 | 31,211 | 131.24 | 5,710 | 5,710 | ±0.2% |
| TensorPlay (bf16) vs Megatron (bf16) | tokens/s | vs vendor |
| --- | ---: | ---: |
| `bf16 + math attn (identical arithmetic)` vs `bf16 + Transformer Engine` | 67,873 vs 65,871 | 1.030× |
| `bf16 + math attn (identical arithmetic)` vs `bf16 + TE + FusedAdam (vendor best)` | 67,873 vs 68,976 | 0.984× |
| `bf16 + math attn (identical arithmetic)` vs `bf16 + unfused math attn` | 67,873 vs 59,538 | 1.140× |
| `bf16 + SDPA attn (production)` vs `bf16 + Transformer Engine` | 77,005 vs 65,871 | 1.169× |
| `bf16 + SDPA attn (production)` vs `bf16 + TE + FusedAdam (vendor best)` | 77,005 vs 68,976 | 1.116× |
| `bf16 + SDPA attn (production)` vs `bf16 + unfused math attn` | 77,005 vs 59,538 | 1.293× |
| `bf16 + SDPA + tp.compile(dynamic=False)` vs `bf16 + Transformer Engine` | 83,883 vs 65,871 | 1.273× |
| `bf16 + SDPA + tp.compile(dynamic=False)` vs `bf16 + TE + FusedAdam (vendor best)` | 83,883 vs 68,976 | 1.216× |
| `bf16 + SDPA + tp.compile(dynamic=False)` vs `bf16 + unfused math attn` | 83,883 vs 59,538 | 1.409× |
| `bf16 + SDPA + compile + fused AdamW (best)` vs `bf16 + Transformer Engine` | 84,247 vs 65,871 | 1.279× |
| `bf16 + SDPA + compile + fused AdamW (best)` vs `bf16 + TE + FusedAdam (vendor best)` | 84,247 vs 68,976 | 1.221× |
| `bf16 + SDPA + compile + fused AdamW (best)` vs `bf16 + unfused math attn` | 84,247 vs 59,538 | 1.415× |
| `fp32 params + bf16 AMP + SDPA` vs `bf16 + Transformer Engine` | 71,171 vs 65,871 | 1.080× |
| `fp32 params + bf16 AMP + SDPA` vs `bf16 + TE + FusedAdam (vendor best)` | 71,171 vs 68,976 | 1.032× |
| `fp32 params + bf16 AMP + SDPA` vs `bf16 + unfused math attn` | 71,171 vs 59,538 | 1.195× |
| TensorPlay fp32 vs Megatron fp32 | speedup |
| --- | ---: |
| `fp32 + math attn` | 1.144× |
wrote bench/results/compare_te.json
exit=0
```
