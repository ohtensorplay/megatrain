"""Probe where time goes in the 160M GPT step, and which knobs actually pay.

Measures (all on one GPU, warmup 2, CUDA-synchronized):
  * forward+backward vs full step           -> optimizer share
  * AdamW default / foreach / fused         -> optimizer kernel choice
  * SDPA backend: flash / cudnn / efficient  -> attention kernel choice

    python bench/opt_probe.py --hidden 1024 --layers 2 --heads 8 \
        --seq 1024 --batch 4 --vocab 65536
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import tensorplay as tp
import tensorplay.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tp_gpt_bench import GPT  # noqa: E402


def timed(fn, iters: int, warmup: int = 2) -> float:
    for _ in range(warmup):
        fn()
    tp.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    tp.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seq", type=int, default=1024)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--hidden", type=int, default=1024)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--vocab", type=int, default=65536)
    p.add_argument("--iters", type=int, default=10)
    args = p.parse_args()

    model = GPT(args.vocab, args.seq, args.hidden, args.layers, args.heads, "sdpa").to("cuda")
    model = model.to(tp.bfloat16)
    tokens = tp.randint(0, args.vocab, (args.batch, args.seq), device="cuda")
    labels = tokens.clone()
    position_ids = tp.arange(args.seq, device="cuda").unsqueeze(0).expand(args.batch, -1)
    mask = tp.ones((1, 1, args.seq, args.seq), dtype=tp.bool, device="cuda").tril()

    token_count = args.batch * args.seq

    def fwd_bwd(opt):
        opt.zero_grad()
        with tp.amp.autocast("cuda", dtype=tp.bfloat16, enabled=True):
            loss = model(tokens, position_ids, mask, labels)
        loss.backward()
        return loss

    names = {
        "adamw-default": dict(),
        "adamw-foreach": dict(foreach=True),
        "adamw-fused": dict(fused=True, capturable=True),
    }
    print(f"geometry={vars(args)}")
    for name, kwargs in names.items():
        try:
            opt = tp.optim.AdamW(model.parameters(), lr=1e-5, **kwargs)
        except Exception as exc:  # noqa: BLE001
            print(f"{name:<18} unsupported: {type(exc).__name__}: {exc}")
            continue
        step_ms = timed(lambda: (fwd_bwd(opt), opt.step()), args.iters)
        fwb_ms = timed(lambda: fwd_bwd(opt), args.iters)
        opt_only = timed(lambda: opt.step(), args.iters)
        print(
            f"{name:<18} step={step_ms:7.2f} ms  fwd+bwd={fwb_ms:7.2f} ms  "
            f"opt={opt_only:5.2f} ms  -> {token_count / (step_ms / 1000):,.0f} tok/s"
        )

    backends = {
        "flash": "FLASH_ATTENTION",
        "cudnn": "CUDNN_ATTENTION",
        "efficient": "EFFICIENT_ATTENTION",
        "math": "MATH",
    }
    opt = tp.optim.AdamW(model.parameters(), lr=1e-5, fused=True, capturable=True)
    enum = tp.nn.attention.SDPBackend
    for name, attr in backends.items():
        member = getattr(enum, attr, None)
        if member is None:
            print(f"sdpa:{name:<12} backend enum missing")
            continue
        try:
            with tp.nn.attention.sdpa_kernel([member]):
                fwb = timed(lambda: fwd_bwd(opt), args.iters)
        except Exception as exc:  # noqa: BLE001
            print(f"sdpa:{name:<12} failed: {type(exc).__name__}: {exc}")
            continue
        print(f"sdpa:{name:<12} fwd+bwd={fwb:7.2f} ms -> {token_count / (fwb / 1000):,.0f} tok/s")


if __name__ == "__main__":
    main()