#!/usr/bin/env python3
# Copyright 2026 TensorPlay Authors
# SPDX-License-Identifier: Apache-2.0
"""System-performance benchmarks for the in-house LoRA variants.

Two independent measurements with a single source of truth (the real
MiniCPM4-0.5B safetensors checkpoint):

  ``--storage`` (CPU, no GPU required)
      Measure the real byte footprint of the frozen base weights: bf16 dense
      versus the packed NF4 representation used by the QLoRA variant. This is
      the weight-residency ("显存") part of system performance and is derived
      from the actual ``quantize_nf4`` output on real checkpoint tensors.

  ``--gpu`` (needs a free GPU)
      Peak VRAM via ``tp.cuda.max_memory_allocated`` and training throughput
      (tokens/s) over a fixed number of real forward+backward steps, for the
      bf16 LoRA baseline and the NF4 QLoRA variant.

Note on references: PEFT / bitsandbytes / torchao are not installed here, so
there is no reference *runtime* to benchmark against. The comparison is
against the in-house bf16 LoRA baseline, and the NF4 layout is byte-for-byte
the upstream NF4 layout (two codes per byte, int8 double-quantized scales),
i.e. the same residency as the reference QLoRA implementations.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import tensorplay as tp

from megatrain.lora.base import LoraConfig, inject_lora
from megatrain.lora.qlora import QLoRAConfig, inject_qlora, quantize_nf4
from megatrain.models.minicpm4 import MiniCPM4ForCausalLM

MODEL = "models/minicpm4-0.5b"
TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def _bytes(t: Any) -> int:
    nbytes = t.nbytes
    return int(nbytes() if callable(nbytes) else nbytes)


def _nf4_bytes(weight: Any, config: QLoRAConfig) -> int:
    state = quantize_nf4(
        weight,
        block_size=config.block_size,
        scaler_block_size=config.scaler_block_size,
        double_quant=config.double_quant,
    )
    return sum(
        _bytes(state[key])
        for key in ("packed", "quantized_scalers", "quantization_factor", "scaler_mean")
    )


def measure_storage(model_path: str) -> dict[str, Any]:
    """Real byte footprint of dense bf16 vs packed NF4, from checkpoint weights."""
    model, report = MiniCPM4ForCausalLM.from_pretrained(model_path, device="cpu", dtype=tp.bfloat16)
    config = QLoRAConfig()
    dense = 0
    nf4 = 0
    layers = 0
    for name, module in model.named_parameters():
        parts = name.split(".")
        if not name.endswith(".weight") or parts[-2] not in TARGETS:
            continue
        dense += _bytes(module.data)
        nf4 += _nf4_bytes(module.data, config)
        layers += 1
    total_params = sum(int(p.numel()) for p in model.parameters())
    result = {
        "source": "real MiniCPM4 safetensors weights",
        "target_weight_tensors": layers,
        "dense_bf16_bytes": dense,
        "nf4_bytes": nf4,
        "weight_reduction_x": round(dense / max(nf4, 1), 3),
        "weight_saved_pct": round(100.0 * (1.0 - nf4 / dense), 2),
        "model_total_params": total_params,
        "model_dense_bf16_mib": round(total_params * 2 / 1024**2, 1),
        "checkpoint_report": report,
    }
    return result


def _make_batch(seq: int, batch: int, device: str) -> Any:
    from megatrain.batch import TrainSample, collate
    from megatrain.losses import TensorBatch

    samples = []
    for _ in range(batch):
        ids = [1] + [7] * (seq - 1)
        samples.append(TrainSample(token_ids=ids, loss_mask=[0] + [1] * (seq - 1)))
    return TensorBatch(collate(samples), device=device)


def measure_gpu(
    model_path: str, variant: str, steps: int, seq: int, batch: int, warmup: int
) -> dict[str, Any]:
    """Peak VRAM and tokens/s over real forward+backward steps."""
    assert tp.cuda.is_available(), "CUDA is required for --gpu"
    from megatrain.losses import sft_loss

    tp.cuda.empty_cache()
    model, _ = MiniCPM4ForCausalLM.from_pretrained(model_path, device="cuda", dtype=tp.bfloat16)
    if variant == "qlora":
        state = inject_qlora(model, QLoRAConfig())
    else:
        state = inject_lora(model, LoraConfig())
    optimizer = tp.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1.0e-4)
    batch_data = _make_batch(seq, batch, "cuda")
    tokens_per_step = batch * seq

    def run(n: int) -> float:
        start = time.perf_counter()
        for _ in range(n):
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_data)
            loss, _ = sft_loss(logits, batch_data)
            loss.backward()
            optimizer.step()
        tp.cuda.synchronize()
        return time.perf_counter() - start

    run(warmup)
    tp.cuda.reset_peak_memory_stats()
    elapsed = run(steps)
    peak = tp.cuda.max_memory_allocated()
    result = {
        "variant": variant,
        "steps": steps,
        "seq": seq,
        "batch": batch,
        "tokens_per_step": tokens_per_step,
        "elapsed_s": round(elapsed, 3),
        "tokens_per_s": round(tokens_per_step * steps / elapsed, 1),
        "peak_vram_mib": round(peak / 1024**2, 1),
        "trainable_params": state.parameter_count,
    }
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=MODEL)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--storage", action="store_true", help="CPU byte-footprint measurement")
    p.add_argument("--gpu", action="store_true", help="GPU peak VRAM + throughput")
    p.add_argument("--variant", choices=("lora", "qlora"), default="lora")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--seq", type=int, default=256)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--warmup", type=int, default=3)
    args = p.parse_args()

    if not args.storage and not args.gpu:
        args.storage = True

    report: dict[str, Any] = {}
    if args.storage:
        report["storage"] = measure_storage(args.model)
    if args.gpu:
        report["gpu"] = [
            measure_gpu(args.model, v, args.steps, args.seq, args.batch, args.warmup)
            for v in ("lora", "qlora")
        ]
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
