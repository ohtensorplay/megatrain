# Copyright 2026 TensorPlay Authors
# SPDX-License-Identifier: Apache-2.0
"""Head-to-head benchmark: megaTrain/TensorPlay vs Megatron-Core.

Both benchmarks use the identical model geometry, dtype, sequence length,
micro-batch, token budget and optimizer. Medians over ``--repeat`` runs are
reported, and every number is a real measurement from this machine.

    python bench/compare.py --steps 12 --repeat 3
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time

GEOMETRY = dict(seq=1024, batch=4, hidden=1024, layers=2, heads=8, vocab=65536)


def _run(cmd: list[str], env: dict[str, str]) -> dict:
    started = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    wall = time.perf_counter() - started
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-12:])
        raise RuntimeError(f"command failed: {' '.join(cmd)}\n{tail}")
    text = proc.stdout
    start = text.rfind("\n{")
    payload = json.loads(text[start + 1 :] if start >= 0 else text[text.find("{") :])
    payload["wall_seconds"] = round(wall, 2)
    return payload


def _te_usable() -> bool:
    try:
        import transformer_engine  # noqa: F401
    except Exception:
        return False
    return True


def build_cases(te: bool, g: dict, preset: str) -> list[tuple[str, str, list[str]]]:
    common = [
        "--steps",
        "{steps}",
        "--seq",
        str(g["seq"]),
        "--batch",
        str(g["batch"]),
        "--hidden",
        str(g["hidden"]),
        "--layers",
        str(g["layers"]),
        "--heads",
        str(g["heads"]),
        "--vocab",
        str(g["vocab"]),
    ]
    base_tp = [sys.executable, "bench/tp_gpt_bench.py", *common]
    base_mg = [sys.executable, "bench/megatron_bench.py", *common]
    out = [
        ("megatron-core", "bf16 + unfused math attn", base_mg + ["--precision", "bf16"]),
        (
            "tensorplay",
            "bf16 + math attn (identical arithmetic)",
            base_tp + ["--precision", "bf16", "--attn", "math"],
        ),
        (
            "tensorplay",
            "bf16 + SDPA attn (production)",
            base_tp + ["--precision", "bf16", "--attn", "sdpa"],
        ),
        (
            "tensorplay",
            "bf16 + SDPA + tp.compile(dynamic=False)",
            base_tp + ["--precision", "bf16", "--attn", "sdpa", "--compile"],
        ),
        (
            "tensorplay",
            "bf16 + SDPA + compile + fused AdamW",
            base_tp + ["--precision", "bf16", "--attn", "sdpa", "--compile", "--fused-adam"],
        ),
        (
            "tensorplay",
            "bf16 + SDPA + fused CE + fused AdamW (best)",
            base_tp
            + ["--precision", "bf16", "--attn", "sdpa", "--eager-ce", "--fused-ce", "--fused-adam"],
        ),
        (
            "tensorplay",
            "bf16 + SDPA + compiled trunk + fused CE + fused AdamW",
            base_tp
            + [
                "--precision",
                "bf16",
                "--attn",
                "sdpa",
                "--compile",
                "--eager-ce",
                "--fused-ce",
                "--fused-adam",
            ],
        ),
        (
            "tensorplay",
            "fp32 params + bf16 AMP + SDPA",
            base_tp + ["--precision", "bf16", "--amp", "--attn", "sdpa"],
        ),
        ("megatron-core", "fp32", base_mg + ["--precision", "fp32"]),
        ("tensorplay", "fp32 + math attn", base_tp + ["--precision", "fp32", "--attn", "math"]),
    ]
    if preset == "bf16":
        out = [c for c in out if "bf16" in c[1] and "AMP" not in c[1]]
    if te:
        out.insert(
            0,
            (
                "megatron-core",
                "bf16 + TE + FusedAdam (vendor best)",
                base_mg + ["--precision", "bf16", "--te", "--fused-adam"],
            ),
        )
        out.insert(
            0,
            (
                "megatron-core",
                "bf16 + Transformer Engine",
                base_mg + ["--precision", "bf16", "--te"],
            ),
        )
    return out


def _free_mib() -> int:
    import torch

    free, _total = torch.cuda.mem_get_info()
    return free // 1024**2


def _gpu_util_pct() -> int | None:
    """Whole-GPU utilization from nvidia-smi, or None if it cannot be read."""
    try:
        raw = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            text=True,
            timeout=10,
        )
        return int(raw.strip().splitlines()[0])
    except Exception:  # noqa: BLE001 - best-effort guard
        return None


def _wait_quiet(min_free_mib: int, budget_s: int = 1800) -> None:
    """Block until the GPU is idle and has room; timings under load are invalid.

    This box is shared with other sessions. A co-tenant that is resident but
    idle is fine (it only reduces free memory); a co-tenant that is *running*
    halves throughput, so require two consecutive idle utilization samples.
    """
    waited = 0
    announced = False
    while True:
        free = _free_mib()
        util = _gpu_util_pct()
        roomy = not min_free_mib or free >= min_free_mib
        idle = util is None or util == 0
        if roomy and idle:
            time.sleep(2)
            util2 = _gpu_util_pct()
            if util2 is None or util2 == 0:
                if announced:
                    print("GPU is quiet again.", flush=True)
                return
        if not announced:
            print(f"GPU busy ({free} MiB free, util {util}%); waiting…", flush=True)
            announced = True
        time.sleep(10)
        waited += 10
        if waited > budget_s:
            raise SystemExit(
                f"GPU stayed busy for {budget_s}s (free {free} MiB, util {util}%). "
                "Timings would be invalid; rerun when the GPU is idle."
            )


def measure(
    stack: str,
    label: str,
    template: list[str],
    steps: int,
    repeat: int,
    env: dict[str, str],
    min_free_mib: int = 0,
) -> dict:
    cmd = [part.replace("{steps}", str(steps)) for part in template]
    runs = []
    for index in range(repeat):
        _wait_quiet(min_free_mib)
        for attempt in range(3):
            try:
                runs.append(_run(cmd, env))
                break
            except RuntimeError as exc:
                # A co-tenant grabbing memory mid-run shows up as OOM; wait for
                # a quiet window and redo the rep instead of recording failure.
                if "out of memory" not in str(exc).lower() or attempt == 2:
                    raise
                print("  OOM (GPU got busy mid-run); retrying after a quiet window…", flush=True)
                time.sleep(20)
        print(f"  run {index + 1}/{repeat}: {runs[-1]['tokens_per_second']:,.0f} tok/s", flush=True)
    tps = [r["tokens_per_second"] for r in runs]
    row = {
        "stack": stack,
        "label": label,
        "params_m": runs[0]["params_m"],
        "tokens_per_second": round(statistics.median(tps), 1),
        # Contention from a co-tenant can only *reduce* throughput, so the best
        # rep is the closest estimate of what this config can do on a quiet GPU.
        "best_tokens_per_second": round(max(tps), 1),
        "step_ms": round(statistics.median(r["step_ms"] for r in runs), 2),
        "peak_memory_mib": round(statistics.median(r["peak_memory_mib"] for r in runs), 1),
        "total_peak_mib": round(
            statistics.median(r.get("total_peak_mib", r["peak_memory_mib"]) for r in runs), 1
        ),
        "loss": runs[0]["loss"],
        "runs_tok_per_s": [round(t, 1) for t in tps],
        "spread_pct": round((max(tps) - min(tps)) / statistics.median(tps) * 100, 1),
        "gpu": runs[0].get("gpu"),
    }
    for key in ("amp", "attn", "transformer_engine", "te_layers"):
        if key in runs[0]:
            row[key] = runs[0][key]
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--json-out", default="bench/results/compare.json")
    parser.add_argument(
        "--min-free-mib",
        type=int,
        default=20000,
        help="abort if another process holds GPU memory (invalid timings)",
    )
    parser.add_argument(
        "--preset",
        choices=["full", "bf16"],
        default="full",
        help="bf16: only the three bf16 cases (large-model runs)",
    )
    for key, value in GEOMETRY.items():
        parser.add_argument(f"--{key}", type=int, default=value)
    args = parser.parse_args()

    env = dict(os.environ)
    env["PYTHONPATH"] = "bench:src"
    te = _te_usable()
    geometry = {k: getattr(args, k) for k in GEOMETRY}
    print(f"transformer_engine available: {te}; geometry {geometry}", flush=True)

    rows = []
    for stack, label, template in build_cases(te, geometry, args.preset):
        row = measure(stack, label, template, args.steps, args.repeat, env, args.min_free_mib)
        rows.append(row)
        print(
            f"{label:<40} {row['tokens_per_second']:>9,.0f} tok/s  "
            f"{row['step_ms']:>7.2f} ms  {row['peak_memory_mib']:>8,.0f} MiB  "
            f"±{row['spread_pct']}%",
            flush=True,
        )

    print(
        "\n| stack | setup | params M | tokens/s | step ms | step peak MiB |"
        " total peak MiB | spread |"
    )
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in rows:
        print(
            f"| {r['stack']} | {r['label']} | {r['params_m']} | "
            f"{r['tokens_per_second']:,.0f} | {r['step_ms']:.2f} | "
            f"{r['peak_memory_mib']:,.0f} | {r['total_peak_mib']:,.0f} | ±{r['spread_pct']}% |"
        )

    vendor = [r for r in rows if r["stack"] == "megatron-core" and "bf16" in r["label"]]
    tp_bf16 = [r for r in rows if r["stack"] == "tensorplay" and "bf16" in r["label"]]
    print("\n| TensorPlay (bf16) vs Megatron (bf16) | tokens/s | vs vendor |")
    print("| --- | ---: | ---: |")
    for tp in tp_bf16:
        for mg in vendor:
            ratio = tp["tokens_per_second"] / mg["tokens_per_second"]
            print(
                f"| `{tp['label']}` vs `{mg['label']}` | "
                f"{tp['tokens_per_second']:,.0f} vs {mg['tokens_per_second']:,.0f} | {ratio:.3f}× |"
            )

    mg_fp32 = next((r for r in rows if r["label"] == "fp32"), None)
    tp_fp32 = next((r for r in rows if r["label"] == "fp32 + math attn"), None)
    if mg_fp32 and tp_fp32:
        ratio = tp_fp32["tokens_per_second"] / mg_fp32["tokens_per_second"]
        print("\n| TensorPlay fp32 vs Megatron fp32 | speedup |")
        print("| --- | ---: |")
        print(f"| `fp32 + math attn` | {ratio:.3f}× |")

    payload = {
        "gpu": rows[0]["gpu"],
        "geometry": geometry,
        "steps_per_run": args.steps,
        "repeat": args.repeat,
        "transformer_engine": te,
        "rows": rows,
    }
    os.makedirs(os.path.dirname(args.json_out), exist_ok=True)
    with open(args.json_out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
