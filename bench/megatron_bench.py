# Copyright 2026 TensorPlay Authors
# SPDX-License-Identifier: Apache-2.0
"""Megatron-Core single-GPU training-core benchmark (NVIDIA baseline).

Based on upstream-checkouts/Megatron-LM/examples/run_simple_mcore_train_loop.py at the
revision pinned in upstream-checkouts/UPSTREAMS.md. It uses Megatron-Core's GPTModel with
TP=PP=1 on one GPU and reports step time / tokens per second / peak memory.

Note: the model architecture is a Megatron GPT (attention + MLP), while
megaTrain's bench uses a tiny embedding+MLP policy. Compare only with this
caveat in mind; it is a representative, not an identical, workload.

    python bench/megatron_bench.py --steps 12 --seq 1024 --batch 4 \
        --hidden 1024 --layers 2 --heads 8 --vocab 65536
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch


def _drop_unusable_te() -> None:
    """A half-installed/cuBLAS-mismatched TE must not break the torch baseline.

    ``import transformer_engine`` raises OSError (not ImportError) when the core
    library cannot be loaded, which Megatron does not catch. Poison the module
    entry so downstream imports fail with ImportError and take the torch path.
    """
    try:
        import transformer_engine  # noqa: F401
    except Exception as exc:  # noqa: BLE001 - any load failure means "unusable"
        print(f"transformer_engine unusable, falling back to torch: {exc}", flush=True)
        sys.modules["transformer_engine"] = None


_drop_unusable_te()

from megatron.core import parallel_state
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig


def _has_te() -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec("transformer_engine") is not None
    except (ImportError, ValueError):
        return False


def init_parallel() -> None:
    if not torch.distributed.is_initialized():
        rank = int(os.environ.get("RANK", "0"))
        world = int(os.environ.get("WORLD_SIZE", "1"))
        torch.cuda.set_device(rank)
        torch.distributed.init_process_group(
            backend="nccl", rank=rank, world_size=world,
            init_method="tcp://127.0.0.1:29501",
        )
    parallel_state.destroy_model_parallel()
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1
    )


def build_model(args):
    model_parallel_cuda_manual_seed(1234)
    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float32
    config = TransformerConfig(
        num_layers=args.layers,
        hidden_size=args.hidden,
        num_attention_heads=args.heads,
        use_cpu_initialization=False,
        # ModelParallelConfig.params_dtype defaults to float32 and is NOT derived
        # from bf16=True, which produces float32 linears + bf16 softmax probs and
        # crashes the unfused bmm. Set it explicitly.
        params_dtype=dtype,
        pipeline_dtype=dtype,
        bf16=args.precision == "bf16",
        # TensorPlay's benchmark model has no dropout; Megatron defaults to 0.1.
        # Matching them is required for a throughput comparison.
        attention_dropout=0.0,
        hidden_dropout=0.0,
        attention_backend="auto" if args.te else "unfused",
    )
    if args.te:
        from megatron.core.models.gpt.gpt_layer_specs import (
            get_gpt_layer_with_transformer_engine_spec,
        )

        layer_spec = get_gpt_layer_with_transformer_engine_spec()
    else:
        layer_spec = get_gpt_layer_local_spec()
    return GPTModel(
        config=config,
        transformer_layer_spec=layer_spec,
        vocab_size=args.vocab,
        max_sequence_length=args.seq,
    ).cuda()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--seq", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--vocab", type=int, default=65536)
    parser.add_argument("--flash", action="store_true", help="use flash attention if available")
    parser.add_argument("--te", action="store_true", help="use Transformer Engine layer spec")
    parser.add_argument("--fused-adam", action="store_true",
                        help="use Transformer Engine FusedAdam (vendor-best optimizer)")
    parser.add_argument("--no-autocast", action="store_true",
                        help="bf16 params without the autocast wrapper")
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="fp32",
                        help="bf16 needs Transformer Engine; fp32 is the no-TE fallback")
    args = parser.parse_args()

    init_parallel()
    model = build_model(args)
    if args.fused_adam:
        # NVIDIA's own fused optimizer: give the vendor stack its best kernel.
        import transformer_engine.pytorch as te

        optimizer = te.optimizers.FusedAdam(model.parameters(), lr=1e-5)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    # GPTModel expects [batch, seq] (current_batch_size = input_ids.shape[0]).
    tokens = torch.randint(0, args.vocab, (args.batch, args.seq), dtype=torch.long, device="cuda")
    labels = tokens.clone()
    position_ids = (
        torch.arange(args.seq, device="cuda").unsqueeze(0).expand(args.batch, -1)
    )
    # Attention scores are [batch, heads, query_seq, key_seq].
    attention_mask = (
        torch.ones((args.batch, 1, 1, args.seq), dtype=torch.bool, device="cuda").tril()
    )

    def step():
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            "cuda", dtype=torch.bfloat16,
            enabled=args.precision == "bf16" and not args.no_autocast,
        ):
            output = model(tokens, position_ids, attention_mask, labels=labels)
        loss = output.float().mean()
        loss.backward()
        optimizer.step()
        return float(loss.detach())

    for _ in range(2):
        step()
    torch.cuda.synchronize()
    # Same two definitions as bench/tp_gpt_bench.py:
    #   peak_memory_mib : process-wide high-water mark
    #   step_peak_mib   : high-water mark of the timed window only
    total_peak_mib = torch.cuda.max_memory_allocated() / 1024**2
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for _ in range(args.steps):
        loss = step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    token_count = args.steps * args.batch * args.seq
    torch.distributed.destroy_process_group()
    print(
        json.dumps(
            {
                "stack": "megatron-core",
                "precision": args.precision,
                "transformer_engine": _has_te(),
                # Hard evidence that TE kernels are actually in the module tree,
                # not merely importable.
                "te_layers": sorted(
                    {
                        type(m).__name__
                        for m in model.modules()
                        if type(m).__module__.startswith("transformer_engine")
                    }
                ),
                "gpu": torch.cuda.get_device_name(0),
                "params_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
                "config": vars(args),
                "loss": round(loss, 5),
                "seconds": round(elapsed, 4),
                "step_ms": round(elapsed / args.steps * 1000, 2),
                "tokens_per_second": round(token_count / elapsed, 1),
                "peak_memory_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
        "total_peak_mib": round(total_peak_mib, 1),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
