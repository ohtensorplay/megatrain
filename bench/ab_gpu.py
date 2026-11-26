"""Reproducible GPU A/B for the megaTrain optimization claims.

Compares a legacy-style step (host-list batch rebuild + full log_softmax CE)
against the optimized paths (zero-copy device batch, logsumexp CE, chunked CE,
AMP). All variants use the same model, batch, seed and hardware, and assert
that the loss is unchanged.

    python bench/ab_gpu.py --steps 15 --seq 1024 --batch 4 --dim 1024 \
        --vocab 65536 --accum 2

This is a core-only microbenchmark. It proves optimization wins *within
megaTrain*; it does NOT prove superiority over Megatron-Core/NeMo-RL, which
need their own stack on the same GPU.
"""

from __future__ import annotations

import argparse
import json
import time

import tensorplay as tp
import tensorplay.nn.functional as F

from megatrain import BaseTrainer, TensorBatch, TrainerConfig, sft_loss

from bench_step import BenchPolicy, make_samples


def legacy_loss(logits, tensors):
    """Old path: materialize the full [B, L, vocab] log_softmax."""
    shift_logits = logits[:, :-1, :]
    shift_ids = tensors.input_ids[:, 1:]
    logps = tp.gather(
        F.log_softmax(shift_logits, dim=-1), -1, shift_ids.unsqueeze(-1)
    ).squeeze(-1)
    mask = tensors.loss_mask[:, 1:]
    loss = -(logps * mask).sum() / mask.sum().clamp(min=1.0)
    return loss, {"ce": loss.detach(), "tokens": mask.sum().detach()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--seq", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--vocab", type=int, default=65536)
    parser.add_argument("--accum", type=int, default=2)
    parser.add_argument("--chunk", type=int, default=4096)
    args = parser.parse_args()

    if not tp.cuda.is_available():
        raise SystemExit("CUDA GPU required; CPU numbers cannot prove anything")
    device = "cuda"
    samples = make_samples(args.vocab, args.seq, args.batch)
    from megatrain import collate

    host_batch = collate(samples)
    device_batch = TensorBatch(host_batch, device=device)
    tokens = args.steps * args.batch * args.seq

    variants = [
        ("legacy", False, "float16", None, legacy_loss, host_batch),
        ("optimized_native", False, "float16", None, sft_loss, device_batch),
        ("optimized_native_zero_copy", False, "float16", "native", sft_loss, device_batch),
        ("optimized_chunked_native", False, "float16", "native", sft_loss, device_batch),
        ("optimized_logsumexp", False, "float16", "logsumexp", sft_loss, device_batch),
        ("optimized_amp_fp16", True, "float16", "native", sft_loss, device_batch),
        ("optimized_amp_bf16", True, "bfloat16", "native", sft_loss, device_batch),
        ("optimized_amp_bf16_ignore", True, "bfloat16", "native_ignore", sft_loss, device_batch),
        ("optimized_amp_bf16_compile", True, "bfloat16", "native", sft_loss, device_batch),
        ("optimized_amp_bf16_fused", True, "bfloat16", "native", sft_loss, device_batch),
        ("optimized_amp_bf16_compile_fused", True, "bfloat16", "native", sft_loss, device_batch),
    ]
    results = []
    for name, amp, amp_dtype, impl, loss_fn, batch in variants:
        tp.manual_seed(1234)
        policy = BenchPolicy(args.vocab, args.dim, args.layers).to(device)
        config = TrainerConfig(
            device=device,
            learning_rate=1e-5,
            gradient_accumulation_steps=args.accum,
            amp=amp,
            amp_dtype=amp_dtype,
            compile_policy=name in (
                "optimized_amp_bf16_compile",
                "optimized_amp_bf16_compile_fused",
            ),
            fused_optimizer=name in (
                "optimized_amp_bf16_fused",
                "optimized_amp_bf16_compile_fused",
            ),
            ce_chunk_tokens=args.chunk if name == "optimized_chunked_native" else None,
            max_grad_norm=1.0,
        )
        trainer = BaseTrainer(policy, config)
        kwargs = {"ce_impl": impl} if impl is not None else {}
        if name == "optimized_chunked_native":
            kwargs["ce_chunk_tokens"] = args.chunk
        for _ in range(2):
            metrics = trainer.train_step(batch, loss_fn=loss_fn, loss_kwargs=kwargs)
        tp.cuda.synchronize()
        tp.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        for _ in range(args.steps):
            metrics = trainer.train_step(batch, loss_fn=loss_fn, loss_kwargs=kwargs)
        tp.cuda.synchronize()
        elapsed = time.perf_counter() - started
        results.append(
            {
                "variant": name,
                "amp": amp,
                "amp_dtype": amp_dtype,
                "ce_impl": impl or "native",
                "loss": round(metrics["train/loss"], 5),
                "seconds": round(elapsed, 4),
                "step_ms": round(elapsed / args.steps * 1000, 2),
                "tokens_per_second": round(tokens / elapsed, 1),
                "peak_memory_mib": round(tp.cuda.max_memory_allocated() / 1024**2, 1),
            }
        )

    baseline_loss = results[0]["loss"]
    for row in results[1:5]:
        assert abs(row["loss"] - baseline_loss) < 1e-3, (
            f"loss mismatch: {row['variant']} {row['loss']} vs legacy {baseline_loss}"
        )
    base = results[0]
    for row in results[1:]:
        row["speedup_vs_legacy"] = round(
            row["tokens_per_second"] / base["tokens_per_second"], 3
        )
        row["peak_mem_saved_mib"] = round(
            base["peak_memory_mib"] - row["peak_memory_mib"], 1
        )
    print(json.dumps({"gpu": tp.cuda.get_device_name(0), "config": vars(args), "results": results}, indent=2))


if __name__ == "__main__":
    main()
