# Copyright 2026 TensorPlay Authors
# SPDX-License-Identifier: Apache-2.0
"""TensorPlay GPT training-core benchmark, architecture-matched to Megatron-Core.

This is the counterpart of ``bench/megatron_bench.py``: same vocabulary, hidden
size, layer count, head count, sequence length, micro-batch, learned absolute
position embeddings, pre-norm blocks, GELU MLP with 4x expansion, untied
embedding/output weights and a mean cross-entropy loss over all tokens.

Two attention implementations are exposed so the comparison can be made
kernel-for-kernel as well as end-to-end:

* ``--attn math``: explicit matmul + causal mask + softmax + matmul. This is the
  same arithmetic Megatron-Core's ``attention_backend=unfused`` path executes.
* ``--attn sdpa``: PyTorch/TensorPlay ``scaled_dot_product_attention``, i.e. the
  kernel megaTrain uses in production.

``--precision bf16`` keeps bf16 weights and runs the forward under bf16
autocast, matching Megatron's ``bf16=True, params_dtype=bfloat16`` setup.
``--amp`` keeps fp32 weights and autocasts to bf16, which is megaTrain's
recommended production path (fp32 master weights + bf16 compute).

    python bench/tp_gpt_bench.py --steps 12 --seq 1024 --batch 4 \
        --hidden 1024 --layers 2 --heads 8 --vocab 65536 \
        --precision bf16 --attn math
"""

from __future__ import annotations

import argparse
import json
import math
import time

import tensorplay as tp
import tensorplay.nn as nn
import tensorplay.nn.functional as F


class Block(nn.Module):
    def __init__(self, hidden: int, heads: int, attn: str) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = hidden // heads
        self.attn = attn
        self.ln1 = nn.LayerNorm(hidden)
        self.qkv = nn.Linear(hidden, 3 * hidden)
        self.proj = nn.Linear(hidden, hidden)
        self.ln2 = nn.LayerNorm(hidden)
        self.fc1 = nn.Linear(hidden, 4 * hidden)
        self.fc2 = nn.Linear(4 * hidden, hidden)

    def attention(self, x, mask):
        batch, seq, hidden = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.reshape(batch, seq, self.heads, self.head_dim).transpose(1, 2)
        k = k.reshape(batch, seq, self.heads, self.head_dim).transpose(1, 2)
        v = v.reshape(batch, seq, self.heads, self.head_dim).transpose(1, 2)
        if self.attn == "sdpa":
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            scores = scores.masked_fill(mask, -10000.0)
            probs = F.softmax(scores, dim=-1)
            out = probs @ v
        out = out.transpose(1, 2).reshape(batch, seq, hidden)
        return self.proj(out)

    def mlp(self, x):
        return self.fc2(F.gelu(self.fc1(x)))

    def forward(self, x, mask):
        x = x + self.attention(self.ln1(x), mask)
        x = x + self.mlp(self.ln2(x))
        return x


def cross_entropy(logits, labels, chunk_tokens: int, fp32: bool = False):
    """Mean CE over all tokens; ``chunk_tokens > 0`` bounds CE workspace size.

    PyTorch's monolithic cross_entropy keeps two full [tokens, vocab] buffers
    alive (log_softmax + its grad); at vocab 65536 that is ~1.3 GiB in bf16.
    Chunking trades a Python loop for that memory.
    """
    vocab = logits.shape[-1]
    if fp32:
        # Megatron's compute_language_model_loss upcasts logits to fp32 before
        # the softmax; doing the same keeps the loss precision matched.
        logits = logits.float()
    logits = logits.reshape(-1, vocab)
    labels = labels.reshape(-1)
    if chunk_tokens <= 0:
        return F.cross_entropy(logits, labels)
    total = logits.shape[0]
    acc = None
    for start in range(0, total, chunk_tokens):
        part = F.cross_entropy(
            logits[start : start + chunk_tokens],
            labels[start : start + chunk_tokens],
            reduction="sum",
        )
        acc = part if acc is None else acc + part
    return acc / total


class GPT(nn.Module):
    def __init__(
        self,
        vocab: int,
        seq: int,
        hidden: int,
        layers: int,
        heads: int,
        attn: str,
        ce_chunk: int = 0,
        fused_ce: bool = False,
        ce_fp32: bool = False,
        eager_ce: bool = False,
    ) -> None:
        super().__init__()
        self.ce_chunk = ce_chunk
        self.fused_ce = fused_ce
        self.ce_fp32 = ce_fp32
        self.eager_ce = eager_ce
        self.tok = nn.Embedding(vocab, hidden)
        self.pos = nn.Embedding(seq, hidden)
        self.blocks = nn.ModuleList([Block(hidden, heads, attn) for _ in range(layers)])
        self.ln_f = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, vocab, bias=False)

    def forward(self, tokens, position_ids, mask, labels):
        x = self.tok(tokens) + self.pos(position_ids)
        for block in self.blocks:
            x = block(x, mask)
        x = self.ln_f(x)
        if self.eager_ce:
            # The loss is applied outside whatever region wraps this forward,
            # so the projection is the loss's own business (Megatron's layout).
            return x
        if self.fused_ce:
            # Fused head+CE: never materializes [tokens, vocab] logits.
            return F.linear_cross_entropy(
                x.reshape(-1, x.shape[-1]), self.head.weight, labels.reshape(-1)
            )
        logits = self.head(x)
        return cross_entropy(logits, labels, self.ce_chunk, self.ce_fp32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--seq", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--vocab", type=int, default=65536)
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--amp", action="store_true", help="fp32 weights + bf16 autocast")
    parser.add_argument(
        "--no-autocast",
        action="store_true",
        help="bf16 weights without the autocast wrapper (fewer dispatches)",
    )
    parser.add_argument(
        "--ce-fp32",
        action="store_true",
        help="upcast logits to fp32 inside CE (matches Megatron's loss)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--attn", choices=["sdpa", "math"], default="math")
    parser.add_argument(
        "--ce-chunk-tokens",
        type=int,
        default=0,
        help="0 = monolithic cross_entropy, >0 = chunked workspace",
    )
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument(
        "--accum",
        type=int,
        default=1,
        help="micro-batches per optimizer step (amortizes optimizer)",
    )
    parser.add_argument("--compile", action="store_true", help="wrap the forward in tp.compile")
    parser.add_argument(
        "--fused-ce",
        action="store_true",
        help="F.linear_cross_entropy (fused head+CE, no logits tensor)",
    )
    parser.add_argument(
        "--eager-ce",
        action="store_true",
        help="return trunk states from the (compiled) forward and "
        "apply head+CE outside it, as Megatron does",
    )
    parser.add_argument("--compile-fullgraph", action="store_true")
    parser.add_argument("--compile-max-autotune", action="store_true")
    parser.add_argument(
        "--cuda-graph",
        action="store_true",
        help="capture the whole step in a CUDA graph (needs capturable Adam)",
    )
    parser.add_argument(
        "--fused-adam", action="store_true", help="tp.optim.AdamW(fused=True, capturable=True)"
    )
    parser.add_argument(
        "--compile-dynamic",
        action="store_true",
        help="tp.compile(dynamic=True), for variable shapes/packed batches",
    )
    args = parser.parse_args()
    tp.manual_seed(args.seed)

    torch_dtype = tp.bfloat16 if args.precision == "bf16" else tp.float32
    model = GPT(
        args.vocab,
        args.seq,
        args.hidden,
        args.layers,
        args.heads,
        args.attn,
        args.ce_chunk_tokens,
        args.fused_ce,
        args.ce_fp32,
        args.eager_ce,
    ).to("cuda")
    if not args.amp:
        model = model.to(torch_dtype)
    adam_kwargs = {"fused": True, "capturable": True} if args.fused_adam else {}
    optimizer = tp.optim.AdamW(model.parameters(), lr=args.lr, **adam_kwargs)

    tokens = tp.randint(0, args.vocab, (args.batch, args.seq), device="cuda")
    labels = tokens.clone()
    position_ids = tp.arange(args.seq, device="cuda").unsqueeze(0).expand(args.batch, -1)
    mask = tp.ones((1, 1, args.seq, args.seq), dtype=tp.bool, device="cuda").tril()

    autocast = (args.amp or args.precision == "bf16") and not args.no_autocast
    if args.compile or args.compile_dynamic:
        run = tp.compile(model, dynamic=args.compile_dynamic, mode="reduce-overhead")
    else:
        run = model

    if args.compile_fullgraph or args.compile_max_autotune:
        run = tp.compile(
            model,
            dynamic=False,
            mode="max-autotune" if args.compile_max_autotune else "reduce-overhead",
            fullgraph=args.compile_fullgraph,
        )

    def head_ce(hidden, target):
        flat = hidden.reshape(-1, args.hidden)
        ids = target.reshape(-1)
        if args.fused_ce:
            return F.linear_cross_entropy(flat, model.head.weight, ids)
        return F.cross_entropy(F.linear(flat, model.head.weight), ids)

    def raw_step():
        optimizer.zero_grad()
        loss = None
        for _ in range(args.accum):
            with tp.amp.autocast("cuda", dtype=tp.bfloat16, enabled=autocast):
                out = run(tokens, position_ids, mask, labels)
                part = head_ce(out, labels) if args.eager_ce else out
            (part / args.accum).backward()
            loss = part if loss is None else loss + part
        optimizer.step()
        return loss / args.accum

    graph = None
    if args.cuda_graph:
        for _ in range(3):
            raw_step()
        tp.cuda.synchronize()
        side = tp.cuda.Stream()
        side.wait_stream(tp.cuda.current_stream())
        with tp.cuda.stream(side):
            for _ in range(2):
                raw_step()
        tp.cuda.current_stream().wait_stream(side)
        tp.cuda.synchronize()
        graph = tp.cuda.CUDAGraph()
        with tp.cuda.graph(graph):
            static_loss = raw_step()
        tp.cuda.synchronize()

        def step():
            graph.replay()
            return float(static_loss.detach())

        loss = step()
    else:

        def step():
            return float(raw_step().detach())

    for _ in range(2):
        step()
    tp.cuda.synchronize()
    # Two memory numbers, same definition on both stacks:
    #   peak_memory_mib : process-wide high-water mark (setup + compile scratch)
    #   step_peak_mib   : high-water mark of the timed window only (after warmup),
    #                     i.e. activations + logits workspace + grad buffers
    total_peak_mib = tp.cuda.max_memory_allocated() / 1024**2
    tp.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(args.steps):
        loss = step()
    tp.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    token_count = args.steps * args.accum * args.batch * args.seq
    print(
        json.dumps(
            {
                "stack": "tensorplay",
                "precision": args.precision,
                "amp": args.amp,
                "attn": args.attn,
                "ce_chunk_tokens": args.ce_chunk_tokens,
                "compile": args.compile or args.compile_dynamic,
                "accum": args.accum,
                "compile_dynamic": args.compile_dynamic,
                "fused_ce": args.fused_ce,
                "eager_ce": args.eager_ce,
                "fused_adam": args.fused_adam,
                "no_autocast": args.no_autocast,
                "ce_fp32": args.ce_fp32,
                "seed": args.seed,
                "cuda_graph": args.cuda_graph,
                "gpu": tp.cuda.get_device_name(0),
                "params_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
                "config": vars(args),
                "loss": round(loss, 5),
                "seconds": round(elapsed, 4),
                "step_ms": round(elapsed / args.steps * 1000, 2),
                "tokens_per_second": round(token_count / elapsed, 1),
                "peak_memory_mib": round(tp.cuda.max_memory_allocated() / 1024**2, 1),
                "total_peak_mib": round(total_peak_mib, 1),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
