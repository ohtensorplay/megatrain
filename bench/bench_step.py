"""Microbenchmark for the megaTrain step loop.

Measures *core overhead* (batching, loss, backward, optimizer) on a small
model. These numbers are NOT comparable to end-to-end NeMo/verl throughput,
which includes generation, reward and multi-node parallelism. On CPU the
result is only a host-overhead smoke test; run on CUDA for real numbers.

Usage:
    python bench/bench_step.py --device cuda --steps 50 --seq 2048 \
        --batch 4 --dim 1024 --vocab 32000 --accum 2
"""

from __future__ import annotations

import argparse
import json
import time

import tensorplay as tp

from megatrain import BaseTrainer, TrainerConfig, TrainSample, collate


class BenchPolicy(tp.nn.Module):
    def __init__(self, vocab: int, dim: int, layers: int) -> None:
        super().__init__()
        self.emb = tp.nn.Embedding(vocab, dim)
        self.blocks = tp.nn.ModuleList([tp.nn.Linear(dim, dim) for _ in range(layers)])
        self.head = tp.nn.Linear(dim, vocab)

    def forward(self, input_ids):
        hidden = self.emb(input_ids)
        for block in self.blocks:
            hidden = block(hidden)
        return self.head(hidden)


def make_samples(vocab: int, seq: int, count: int):
    return [
        TrainSample(
            token_ids=[index % vocab for index in range(seq)],
            loss_mask=[0] * (seq // 2) + [1] * (seq - seq // 2),
        )
        for _ in range(count)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seq", type=int, default=512)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--vocab", type=int, default=4096)
    parser.add_argument("--accum", type=int, default=1)
    parser.add_argument("--chunk", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--ce-impl", default="native", choices=["native", "logsumexp"])
    args = parser.parse_args()

    device = (
        "cuda"
        if args.device == "cuda" or (args.device == "auto" and tp.cuda.is_available())
        else "cpu"
    )
    if device == "cpu" and args.amp:
        raise SystemExit("AMP is disabled on CPU; drop --amp or use --device cuda")
    tp.manual_seed(1)
    policy = BenchPolicy(args.vocab, args.dim, args.layers)
    config = TrainerConfig(
        device=device,
        learning_rate=1e-5,
        gradient_accumulation_steps=args.accum,
        amp=args.amp,
        ce_chunk_tokens=args.chunk or None,
        max_grad_norm=1.0,
    )
    loss_kwargs = {"ce_impl": args.ce_impl}
    trainer = BaseTrainer(policy, config)
    batch = collate(make_samples(args.vocab, args.seq, args.batch))

    def sync() -> None:
        if device == "cuda":
            tp.cuda.synchronize()

    for _ in range(2):
        trainer.train_step(batch, loss_kwargs=loss_kwargs)
    sync()
    if device == "cuda":
        tp.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for _ in range(args.steps):
        trainer.train_step(batch, loss_kwargs=loss_kwargs)
    sync()
    elapsed = time.perf_counter() - started

    tokens = args.steps * args.batch * args.seq
    result = {
        "device": device,
        "steps": args.steps,
        "seq": args.seq,
        "batch": args.batch,
        "dim": args.dim,
        "layers": args.layers,
        "vocab": args.vocab,
        "accum": args.accum,
        "ce_chunk_tokens": config.ce_chunk_tokens,
        "ce_impl": args.ce_impl,
        "amp": config.amp,
        "seconds": round(elapsed, 4),
        "step_seconds": round(elapsed / args.steps, 6),
        "tokens_per_second": round(tokens / elapsed, 1),
        "note": "core-only microbenchmark; not comparable to end-to-end RL throughput",
    }
    if device == "cuda":
        result["peak_memory_bytes"] = int(tp.cuda.max_memory_allocated())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
