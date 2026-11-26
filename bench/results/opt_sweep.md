# 160.46M 极致优化旋钮矩阵（RTX 3090，seq 1024，batch 4，10 step × 3 次）

```bash
bash bench/opt_sweep.sh
```

守卫：每次计时前要求 GPU `util=0` 且 `free≥18 GiB`；进程结束后空闲显存下降 >2 GiB 视为被同机租户扰动，丢弃重测。

> 本文件是 **CE 补丁之前** 的日志。其中 `### --fused-ce --compile` 组跑的是当时尚未融合的
> `F.linear_cross_entropy`（参考实现，且编译区内现在会自动退回参考路径），所以它 ≈ `--compile`，
> 不代表当前行为；融合 CE 的数据见 [`fused_ce.md`](fused_ce.md) 与 `devdocs/performance.md` §4.6。

`step MiB` = 计时窗口峰值（激活 + logits workspace + 梯度缓冲）；`total MiB` = 进程级高水位（含 compile 的一次性 scratch、CUDA graph pool 等常驻部分）。

```
geometry batch=4 steps=10 reps=3 min_free=18000MiB
### (baseline)
#   tok/s    step ms  step MiB  total MiB
  76,983	  53.21	   4280	   4280
  77,815	  52.64	   4280	   4280
  77,299	  52.99	   4280	   4280
### --compile
#   tok/s    step ms  step MiB  total MiB
  FAILED
  84,645	  48.39	   3070	   4606
  84,830	  48.28	   3070	   4606
  FAILED
  83,618	  48.98	   3070	   4606
### --compile-max-autotune
#   tok/s    step ms  step MiB  total MiB
  84,568	  48.43	   3070	   4606
  84,075	  48.72	   3070	   4606
  83,830	  48.86	   3070	   4606
### --compile-fullgraph
#   tok/s    step ms  step MiB  total MiB
  84,350	  48.56	   3070	   4606
  82,995	  49.35	   3070	   4606
  84,297	  48.59	   3070	   4606
### --fused-ce --compile
#   tok/s    step ms  step MiB  total MiB
  FAILED
  83,166	  49.25	   3070	   4606
  FAILED
  83,883	  48.83	   3070	   4606
  84,109	  48.70	   3070	   4606
### --cuda-graph --fused-adam
#   tok/s    step ms  step MiB  total MiB
  78,429	  52.23	   1225	   4280
  78,320	  52.30	   1225	   4280
  78,377	  52.26	   1225	   4280
### --fused-adam --compile
#   tok/s    step ms  step MiB  total MiB
  84,539	  48.45	   3070	   4606
  84,796	  48.30	   3070	   4606
  84,701	  48.36	   3070	   4606
### --accum 4 --compile
#   tok/s    step ms  step MiB  total MiB
  FAILED
  87,976	 186.23	   3070	   4489
  88,183	 185.80	   3070	   4489
  88,394	 185.35	   3070	   4489
### --accum 4
#   tok/s    step ms  step MiB  total MiB
  80,139	 204.45	   4586	   4586
  80,120	 204.49	   4586	   4586
  80,158	 204.40	   4586	   4586
### --no-autocast --ce-fp32 --compile
#   tok/s    step ms  step MiB  total MiB
  84,038	  48.74	   3070	   5513
  FAILED
  84,062	  48.73	   3070	   5513
  84,282	  48.60	   3070	   5513
### --no-autocast --compile
#   tok/s    step ms  step MiB  total MiB
  87,967	  46.56	   2558	   3334
  88,056	  46.52	   2558	   3334
  (discarded a disturbed window)
  88,700	  46.18	   2558	   3334
```
