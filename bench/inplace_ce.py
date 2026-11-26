"""In-place cross entropy: never keeps a second [tokens, vocab] buffer alive.

SUPERSEDED: this Python prototype was the first look at the same idea and
measured *slower* than the reference (it still pays several full-size
elementwise passes). The production version is the Triton kernel in
``tensorplay/nn/_fused_ce.py`` (see ``bench/fused_ce.sh`` and
``bench/results/fused_ce.md``). Kept only as the record of that prototype.

Megatron's ``VocabParallelCrossEntropy`` does this (mutates the logits buffer,
recomputes softmax from it in backward). PyTorch's ``cross_entropy`` instead
keeps logits + log_softmax + grad_input, i.e. ~3x the logits footprint, which at
vocab 65536 dominates memory and hurts larger batches.

This prototype checks numerics against ``F.cross_entropy`` and then measures
memory/throughput at several batch sizes.
"""

from __future__ import annotations

import time

import tensorplay as tp
import tensorplay.nn.functional as F


class InplaceCE(tp.autograd.Function):
    @staticmethod
    def forward(ctx, logits, target):
        rows, _vocab = logits.shape
        shifted = logits - logits.max(dim=-1, keepdim=True).values
        target_logit = shifted.gather(1, target.reshape(-1, 1)).reshape(-1)
        probs = shifted.exp_()  # shifted buffer becomes exp(x - max)
        denom = probs.sum(dim=-1)
        loss = (denom.log() - target_logit).mean()
        ctx.save_for_backward(probs, denom, target)
        ctx.mark_dirty(logits)
        ctx.rows = rows
        return loss

    @staticmethod
    def backward(ctx, grad_out):
        probs, denom, target = ctx.saved_tensors
        grad = probs
        grad.div_(denom.unsqueeze(-1))
        idx = tp.arange(ctx.rows, device=target.device)
        grad[idx, target] -= 1.0
        grad.mul_(grad_out / ctx.rows)
        return grad, None


def main() -> None:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from tp_gpt_bench import GPT

    seq, batch, hidden, layers, heads, vocab = 1024, 4, 1024, 2, 8, 65536
    model = GPT(vocab, seq, hidden, layers, heads, "sdpa").to("cuda").to(tp.bfloat16)
    tokens = tp.randint(0, vocab, (batch, seq), device="cuda")
    labels = tokens.clone()
    pos = tp.arange(seq, device="cuda").unsqueeze(0).expand(batch, -1)
    mask = tp.ones((1, 1, seq, seq), dtype=tp.bool, device="cuda").tril()

    x = model.ln_f(
        (lambda h: h)(  # trunk output, detached so only head/CE differ
            (lambda h0: [b(h0, mask) for b in model.blocks][-1])(
                model.tok(tokens) + model.pos(pos)
            )
        )
    ).detach().requires_grad_(True)
    weight = model.head.weight
    target = labels.reshape(-1)

    def ref():
        logits = F.linear(x, weight)
        return F.cross_entropy(logits.reshape(-1, vocab), target)

    def fused():
        logits = F.linear(x, weight)
        return InplaceCE.apply(logits.reshape(-1, vocab), target)

    lr = ref()
    lr.backward()
    gx_ref = x.grad.detach().float().clone()
    gw_ref = weight.grad.detach().float().clone()
    x.grad = None
    weight.grad = None

    lf = fused()
    lf.backward()
    gx = x.grad.detach().float()
    gw = weight.grad.detach().float()

    def rel(a, b):
        return float((a - b).abs().max() / b.abs().max().clamp_min(1e-6))

    print(f"loss ref   = {float(lr.detach()):.6f}")
    print(f"loss fused = {float(lf.detach()):.6f}")
    print(f"rel err dx = {rel(gx, gx_ref):.3e}  dW = {rel(gw, gw_ref):.3e}")

    def timed(fn, n=6, w=3):
        for _ in range(w):
            fn()
        tp.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        tp.cuda.synchronize()
        return (time.perf_counter() - t0) / n * 1000

    def fwd_bwd(loss_fn):
        if x.grad is not None:
            x.grad = None
            weight.grad = None
        loss = loss_fn()
        loss.backward()
        return loss

    for name, fn in (("cross_entropy", ref), ("inplace_ce", fused)):
        tp.cuda.synchronize()
        tp.cuda.reset_peak_memory_stats()
        ms = timed(lambda: fwd_bwd(fn))
        print(f"{name:<14} head+ce fwd+bwd {ms:6.2f} ms  peak {tp.cuda.max_memory_allocated()/1024**2:7.0f} MiB")


if __name__ == "__main__":
    main()