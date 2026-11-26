#!/usr/bin/env python3
# Copyright 2026 TensorPlay Authors
# SPDX-License-Identifier: Apache-2.0
"""Real LoRA SFT of MiniCPM4-0.5B using only the in-house TensorPlay stack.

Data: Stanford Alpaca's 52k real instruction pairs as plain JSON (stdlib urllib
only, no datasets-server rate limits). Model/config/RoPE: megaLM's definitions.
Training: megaTrain BaseTrainer + sft_loss. Adapter export: PEFT-compatible
safetensors readable by megaLM's LoRA loader.
"""

from __future__ import annotations

import argparse
import json
import random
import urllib.request
from pathlib import Path

import tensorplay as tp
from megalm.runtime.tokenizer import ModelTokenizer

from megatrain import (
    BaseTrainer,
    TrainerConfig,
    TrainSample,
    collate,
    sft_loss,
)
from megatrain.lora.base import LoraConfig, inject_lora
from megatrain.lora.dora import DoRAConfig, inject_dora
from megatrain.lora.qlora import QLoRAConfig, inject_qlora
from megatrain.lora.variants import lora_plus_optimizer
from megatrain.models.minicpm4 import MiniCPM4ForCausalLM

MODEL = "models/minicpm4-0.5b"
DATA_URL = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"


def fetch_rows(max_samples: int, cache: Path) -> list[dict]:
    if cache.exists():
        rows = json.loads(cache.read_text(encoding="utf-8"))
        print(f"[data] loaded {len(rows)} cached rows from {cache}", flush=True)
        return rows
    with urllib.request.urlopen(DATA_URL, timeout=120) as response:
        payload = response.read()
    rows = json.loads(payload)[:max_samples]
    print(f"[data] downloaded {len(rows)} real instruction rows", flush=True)
    cache.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    return rows


def build_samples(rows: list[dict], tok: ModelTokenizer, max_length: int) -> list[TrainSample]:
    samples: list[TrainSample] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        instruction = str(row.get("instruction") or "").strip()
        context = str(row.get("input") or "").strip()
        assistant = str(row.get("output") or "").strip()
        if not instruction or len(assistant) < 10:
            continue
        user = instruction + ("\n\n" + context if context else "")
        key = (user[:80], assistant[:80])
        if key in seen:
            continue
        seen.add(key)
        prompt = list(tok.encode_chat([{"role": "user", "content": user}]))
        full = list(
            tok.encode_chat(
                [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]
            )
        )
        # MiniCPM's chat template renders the assistant-header in the prompt.
        if full[: len(prompt)] != prompt:
            continue
        if len(full) > max_length or len(full) - len(prompt) < 8:
            continue
        samples.append(
            TrainSample(
                token_ids=list(full),
                loss_mask=[0] * len(prompt) + [1] * (len(full) - len(prompt)),
            )
        )
    print(
        f"[data] built {len(samples)} SFT samples "
        f"(mean tokens={sum(s.length for s in samples) / max(1, len(samples)):.0f})",
        flush=True,
    )
    return samples


def evaluate(model, trainer: BaseTrainer, samples: list[TrainSample]) -> float:
    total_loss = 0.0
    total_tokens = 0
    with tp.no_grad():
        for start in range(0, len(samples), 8):
            batch = trainer.tensor_batch_cls(collate(samples[start : start + 8]), device="cuda")
            logits = trainer.forward_fn(model, batch)
            loss, diag = sft_loss(logits, batch)
            tokens = int(diag["tokens"].flatten()[0])
            total_loss += float(loss.flatten()[0]) * tokens
            total_tokens += tokens
    return total_loss / max(total_tokens, 1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=MODEL)
    p.add_argument(
        "--output", default=str(Path(__file__).parent / "runs" / "minicpm4-ultrachat-lora")
    )
    p.add_argument("--max-samples", type=int, default=20000)
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-tokens", type=int, default=2048)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--r", type=int, default=16)
    p.add_argument("--alpha", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--qlora",
        action="store_true",
        help="Use the QLoRA variant: frozen NF4 base + bf16 LoRA adapters.",
    )
    p.add_argument("--dora", action="store_true", help="Use the DoRA variant")
    p.add_argument("--block-size", type=int, default=64)
    p.add_argument("--scaler-block-size", type=int, default=16)
    p.add_argument("--no-double-quant", action="store_true")
    p.add_argument("--rslora", action="store_true", help="Use alpha/sqrt(r) scaling")
    p.add_argument(
        "--lora-plus-ratio",
        type=float,
        default=0.0,
        help="LoRA+ B-matrix LR multiplier (0 disables; reference uses ~16)",
    )
    args = p.parse_args()
    # QLoRA follows the reference recipe's higher adapter learning rate.
    lr = args.lr if args.lr is not None else (2.0e-4 if args.qlora else 5.0e-5)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[setup] {vars(args)}", flush=True)

    tok = ModelTokenizer(args.model)
    rows = fetch_rows(args.max_samples, out / "ultrachat_rows.json")
    samples = build_samples(rows, tok, args.max_length)
    random.Random(args.seed).shuffle(samples)
    split = max(1, int(len(samples) * 0.05))
    train, ev = samples[split:], samples[:split]
    print(f"[data] train={len(train)} eval={len(ev)}", flush=True)

    print("[model] loading MiniCPM4 via megaLM config + TensorPlay weights...", flush=True)
    model, report = MiniCPM4ForCausalLM.from_pretrained(
        args.model, device="cuda", dtype=tp.bfloat16
    )
    print(f"[model] {report}", flush=True)
    targets = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    if args.dora:
        lora = inject_dora(
            model,
            DoRAConfig(
                r=args.r,
                lora_alpha=args.alpha,
                target_modules=targets,
                use_rslora=args.rslora,
            ),
        )
        label = "dora"
    elif args.qlora:
        config = QLoRAConfig(
            r=args.r,
            lora_alpha=args.alpha,
            target_modules=targets,
            block_size=args.block_size,
            scaler_block_size=args.scaler_block_size,
            double_quant=not args.no_double_quant,
            use_rslora=args.rslora,
        )
        lora = inject_qlora(model, config)
        label = "qlora"
    else:
        lora = inject_lora(
            model,
            LoraConfig(
                r=args.r,
                lora_alpha=args.alpha,
                target_modules=targets,
                use_rslora=args.rslora,
            ),
        )
        label = "lora"
    print(
        f"[{label}] {len(lora.layers)} layers, {lora.parameter_count} trainable params",
        flush=True,
    )

    optimizer = None
    if args.lora_plus_ratio > 0:
        optimizer = lora_plus_optimizer(lora, lr=lr, ratio=args.lora_plus_ratio, fused=True)
        print(
            f"[lora+] A lr={lr:g}, B lr={lr * args.lora_plus_ratio:g}",
            flush=True,
        )

    trainer = BaseTrainer(
        model,
        TrainerConfig(
            device="cuda",
            learning_rate=lr,
            gradient_accumulation_steps=args.grad_accum,
            max_grad_norm=1.0,
            amp=True,
            amp_dtype="bfloat16",
            seed=args.seed,
            checkpoint_dir=str(out / "checkpoints"),
        ),
        optimizer=optimizer,
        forward_fn=lambda m, batch: m(batch),
    )

    metrics: dict = {
        "args": {**vars(args), "lr": lr},
        "variant": label,
        "n_train": len(train),
        "n_eval": len(ev),
        "lora_params": lora.parameter_count,
    }
    metrics["eval_loss_before"] = evaluate(model, trainer, ev)
    print(f"[eval] before = {metrics['eval_loss_before']:.4f}", flush=True)

    history: list[dict] = []
    for step_metrics in trainer.fit(
        train,
        max_tokens=args.batch_tokens,
        epochs=args.epochs,
        on_step=lambda m: history.append({k: float(v) for k, v in m.items()}),
    ):
        if trainer.step_count % 25 == 0:
            print(
                f"[train] step={trainer.step_count} "
                f"loss={step_metrics['train/loss']:.4f} "
                f"grad={step_metrics.get('train/grad_norm', float('nan')):.3f}",
                flush=True,
            )
    metrics["eval_loss_after"] = evaluate(model, trainer, ev)
    metrics["history"] = history
    print(f"[eval] after  = {metrics['eval_loss_after']:.4f}", flush=True)

    adapter = out / "adapter"
    lora.export(adapter)
    print(f"[save] adapter -> {adapter}", flush=True)
    (out / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("[done] metrics written", flush=True)


if __name__ == "__main__":
    main()
