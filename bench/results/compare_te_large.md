# 474.07M 参数、seq 2048，**已装 Transformer Engine**（RTX 3090，8 step × 3 次中位数）

```bash
python bench/compare.py --steps 8 --repeat 3 --preset bf16 --seq 2048 --batch 2 \
    --hidden 2048 --layers 4 --heads 16 --vocab 65536

```

注：`(best)` 指该行是生产配置；其中 fused AdamW 相对纯 `tp.compile` 的差异在噪声内（160M: +0.4%，474M: +0.6%），增益来自 `tp.compile`（固定 shape）。两种写法都保留在表里，便于核对。

```
transformer_engine available: True; geometry {'seq': 2048, 'batch': 2, 'hidden': 2048, 'layers': 4, 'heads': 16, 'vocab': 65536}
  run 1/3: 23,569 tok/s
  run 2/3: 23,531 tok/s
  run 3/3: 23,555 tok/s
bf16 + Transformer Engine                   23,555 tok/s   173.89 ms     5,382 MiB  ±0.2%
  run 1/3: 24,715 tok/s
  run 2/3: 24,729 tok/s
  run 3/3: 24,719 tok/s
bf16 + TE + FusedAdam (vendor best)         24,719 tok/s   165.70 ms     7,175 MiB  ±0.1%
  run 1/3: 17,462 tok/s
  run 2/3: 17,454 tok/s
  OOM (GPU got busy mid-run); retrying after a quiet window…
  run 3/3: 17,326 tok/s
bf16 + unfused math attn                    17,454 tok/s   234.67 ms     8,930 MiB  ±0.8%
  run 1/3: 19,668 tok/s
  run 2/3: 19,705 tok/s
GPU busy (23847 MiB free, util 100%); waiting…
GPU is quiet again.
  run 3/3: 19,664 tok/s
bf16 + math attn (identical arithmetic)     19,668 tok/s   208.26 ms     9,917 MiB  ±0.2%
GPU busy (23847 MiB free, util 60%); waiting…
GPU is quiet again.
  run 1/3: 26,447 tok/s
  run 2/3: 26,456 tok/s
  run 3/3: 26,394 tok/s
bf16 + SDPA attn (production)               26,447 tok/s   154.88 ms     6,910 MiB  ±0.2%
GPU busy (23847 MiB free, util 100%); waiting…
GPU is quiet again.
  run 1/3: 26,558 tok/s
  run 2/3: 26,561 tok/s
  run 3/3: 26,533 tok/s
bf16 + SDPA + tp.compile(dynamic=False)     26,558 tok/s   154.23 ms     6,707 MiB  ±0.1%
  run 1/3: 26,555 tok/s
  run 2/3: 26,719 tok/s
  run 3/3: 26,741 tok/s
bf16 + SDPA + compile + fused AdamW (best)    26,719 tok/s   153.30 ms     6,707 MiB  ±0.7%
| stack | setup | params M | tokens/s | step ms | step peak MiB | total peak MiB | spread |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| megatron-core | bf16 + Transformer Engine | 474.07 | 23,555 | 173.89 | 5,382 | 5,382 | ±0.2% |
| megatron-core | bf16 + TE + FusedAdam (vendor best) | 474.07 | 24,719 | 165.70 | 7,175 | 7,175 | ±0.1% |
| megatron-core | bf16 + unfused math attn | 474.07 | 17,454 | 234.67 | 8,930 | 8,930 | ±0.8% |
| tensorplay | bf16 + math attn (identical arithmetic) | 474.07 | 19,668 | 208.26 | 9,917 | 9,917 | ±0.2% |
| tensorplay | bf16 + SDPA attn (production) | 474.07 | 26,447 | 154.88 | 6,910 | 6,910 | ±0.2% |
| tensorplay | bf16 + SDPA + tp.compile(dynamic=False) | 474.07 | 26,558 | 154.23 | 6,707 | 8,243 | ±0.1% |
| tensorplay | bf16 + SDPA + compile + fused AdamW (best) | 474.07 | 26,719 | 153.30 | 6,707 | 8,243 | ±0.7% |
| TensorPlay (bf16) vs Megatron (bf16) | tokens/s | vs vendor |
| --- | ---: | ---: |
| `bf16 + math attn (identical arithmetic)` vs `bf16 + Transformer Engine` | 19,668 vs 23,555 | 0.835× |
| `bf16 + math attn (identical arithmetic)` vs `bf16 + TE + FusedAdam (vendor best)` | 19,668 vs 24,719 | 0.796× |
| `bf16 + math attn (identical arithmetic)` vs `bf16 + unfused math attn` | 19,668 vs 17,454 | 1.127× |
| `bf16 + SDPA attn (production)` vs `bf16 + Transformer Engine` | 26,447 vs 23,555 | 1.123× |
| `bf16 + SDPA attn (production)` vs `bf16 + TE + FusedAdam (vendor best)` | 26,447 vs 24,719 | 1.070× |
| `bf16 + SDPA attn (production)` vs `bf16 + unfused math attn` | 26,447 vs 17,454 | 1.515× |
| `bf16 + SDPA + tp.compile(dynamic=False)` vs `bf16 + Transformer Engine` | 26,558 vs 23,555 | 1.127× |
| `bf16 + SDPA + tp.compile(dynamic=False)` vs `bf16 + TE + FusedAdam (vendor best)` | 26,558 vs 24,719 | 1.074× |
| `bf16 + SDPA + tp.compile(dynamic=False)` vs `bf16 + unfused math attn` | 26,558 vs 17,454 | 1.522× |
| `bf16 + SDPA + compile + fused AdamW (best)` vs `bf16 + Transformer Engine` | 26,719 vs 23,555 | 1.134× |
| `bf16 + SDPA + compile + fused AdamW (best)` vs `bf16 + TE + FusedAdam (vendor best)` | 26,719 vs 24,719 | 1.081× |
| `bf16 + SDPA + compile + fused AdamW (best)` vs `bf16 + unfused math attn` | 26,719 vs 17,454 | 1.531× |
wrote bench/results/compare_te_large.json
exit=0
```
