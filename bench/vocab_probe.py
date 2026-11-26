"""Where the vocabulary projection goes, with and without the fused loss.

At vocab 65536 the output projection and its cross entropy are the largest
single item of a step, so this probe separates them:

    trunk            embedding + blocks + final LayerNorm
    head             hidden -> vocab matmul
    ce               cross entropy over the resulting logits
    head+ce          both, forward and forward+backward

and reports the same split for ``F.linear_cross_entropy``, which never
materializes a second full-size buffer.

    python bench/vocab_probe.py --hidden 1024 --layers 2 --seq 1024 --batch 4
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


def timed(fn, iters: int = 10, warmup: int = 3) -> float:
    for _ in range(warmup):
        fn()
    tp.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    tp.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


def peak_mib(fn) -> float:
    tp.cuda.synchronize()
    tp.cuda.empty_cache()
    tp.cuda.reset_peak_memory_stats()
    fn()
    tp.cuda.synchronize()
    return tp.cuda.max_memory_allocated() / 1024 ** 2


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

    model = GPT(args.vocab, args.seq, args.hidden, args.layers, args.heads,
                "sdpa", eager_ce=True).to("cuda").to(tp.bfloat16)
    ids = tp.randint(0, args.vocab, (args.batch, args.seq), device="cuda")
    pos = tp.arange(args.seq, device="cuda").unsqueeze(0).expand(args.batch, -1)
    mask = tp.ones((1, 1, args.seq, args.seq), dtype=tp.bool, device="cuda").tril()
    head = model.head.weight
    amp = tp.amp.autocast("cuda", dtype=tp.bfloat16, enabled=True)
    rows = args.batch * args.seq
    flat_ids = ids.reshape(-1)

    def trunk():
        return model(ids, pos, mask, ids)

    hidden = trunk().detach()
    print(f"geometry={vars(args)}  tokens={rows}")

    with amp:
        hidden = trunk()
        logits = F.linear(hidden.reshape(-1, args.hidden), head)
        z = logits.detach().requires_grad_(True)

    def head_fwd():
        with amp:
            return F.linear(hidden.reshape(-1, args.hidden), head)

    def head_fb():
        head.zero_grad() if hasattr(head, "zero_grad") else None
        h = hidden.reshape(-1, args.hidden).detach().requires_grad_(True)
        with amp:
            out = F.linear(h, head)
        out.sum().backward()

    def ce_fwd():
        with amp:
            return F.cross_entropy(logits, flat_ids)

    def ce_fb():
        buf = z
        buf.grad = None
        with amp:
            loss = F.cross_entropy(buf, flat_ids)
        loss.backward()

    def fused_fwd():
        h = hidden.reshape(-1, args.hidden).detach().requires_grad_(True)
        with amp:
            return F.linear_cross_entropy(h, head, flat_ids)

    def fused_fb():
        h = hidden.reshape(-1, args.hidden).detach().requires_grad_(True)
        with amp:
            loss = F.linear_cross_entropy(h, head, flat_ids)
        loss.backward()

    trunk_f, head_f, ce_f = timed(trunk), timed(head_fwd), timed(ce_fwd)
    head_fb_ms, ce_fb_ms = timed(head_fb), timed(ce_fb)
    fused_f, fused_fb_ms = timed(fused_fwd), timed(fused_fb)

    print(f"\n{'segment':<26}{'fwd ms':>9}{'fwd+bwd ms':>12}{'peak MiB':>11}")
    print(f"{'trunk':<26}{trunk_f:>9.2f}{'-':>12}")
    print(f"{'head (hidden->vocab)':<26}{head_f:>9.2f}{head_fb_ms:>12.2f}")
    print(f"{'cross_entropy (logits)':<26}{ce_f:>9.2f}{ce_fb_ms:>12.2f}"
          f"{peak_mib(ce_fb):>11.0f}")
    print(f"{'head+ce reference':<26}{head_f + ce_f:>9.2f}{head_fb_ms + ce_fb_ms:>12.2f}")
    print(f"{'head+ce fused':<26}{fused_f:>9.2f}{fused_fb_ms:>12.2f}"
          f"{peak_mib(fused_fb):>11.0f}")
    print(f"\nforward total: {(trunk_f + head_f + ce_f):.2f} ms -> "
          f"{(trunk_f + fused_f):.2f} ms fused")
    print(f"ce share of forward: {(head_f + ce_f) / (trunk_f + head_f + ce_f) * 100:.0f}% "
          f"-> {fused_f / (trunk_f + fused_f) * 100:.0f}% fused")
    print(f"head+ce fwd+bwd: {head_fb_ms + ce_fb_ms:.2f} -> {fused_fb_ms:.2f} ms "
          f"({(head_fb_ms + ce_fb_ms) / fused_fb_ms:.2f}x)")


if __name__ == "__main__":
    main()