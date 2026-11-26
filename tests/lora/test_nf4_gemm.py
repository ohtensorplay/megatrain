# Copyright 2026 TensorPlay Authors
# SPDX-License-Identifier: Apache-2.0
"""Fused NF4 GEMM: forward/backward parity and no dense weight is retained."""

import pytest
import tensorplay as tp
import tensorplay.nn.functional as F
from megalm.kernels.nf4_gemm import (
    nf4_gemm_backward,
    nf4_gemm_forward,
    nf4_linear,
    nf4_linear_scratch,
)

from megatrain.lora.qlora import QLoRAConfig, dequantize_nf4, quantize_nf4

pytestmark = pytest.mark.skipif(not tp.cuda.is_available(), reason="fused NF4 GEMM requires CUDA")


def _state(n: int, k: int):
    tp.manual_seed(21)
    weight = tp.randn((n, k), device="cuda", dtype=tp.bfloat16) * 0.3
    state = quantize_nf4(weight, block_size=64, scaler_block_size=16, double_quant=True)
    return state, QLoRAConfig.codebook(tp.float32, "cuda")


def _call_forward(state, codebook, x, *, fused=True):
    args = (
        state["packed"],
        state["quantized_scalers"],
        state["quantization_factor"],
        state["scaler_mean"],
        codebook,
    )
    kwargs = dict(block=state["block_size"], sb=state["scaler_block_size"])
    return (
        nf4_gemm_forward(x, *args, **kwargs)
        if fused
        else F.linear(x, dequantize_nf4(state, dtype=tp.bfloat16))
    )


def test_nf4_forward_matches_dequant_gemm():
    state, codebook = _state(512, 1024)
    x = tp.randn((64, 1024), device="cuda", dtype=tp.bfloat16)
    got = _call_forward(state, codebook, x)
    ref = _call_forward(state, codebook, x, fused=False)
    # bf16 accumulation differs by at most one ULP at this magnitude.
    assert float((got - ref).abs().max()) <= 0.25


def test_nf4_backward_matches_dequant_gemm():
    state, codebook = _state(512, 1024)
    grad = tp.randn((64, 512), device="cuda", dtype=tp.bfloat16)
    got = nf4_gemm_backward(
        grad,
        state["packed"],
        state["quantized_scalers"],
        state["quantization_factor"],
        state["scaler_mean"],
        codebook,
        block=state["block_size"],
        sb=state["scaler_block_size"],
    )
    dense = dequantize_nf4(state, dtype=tp.bfloat16)
    ref = grad @ dense
    assert float((got - ref).abs().max()) < 0.01


def test_nf4_linear_saves_no_dense_weight():
    state, codebook = _state(256, 512)
    x = tp.randn((32, 512), device="cuda", dtype=tp.bfloat16, requires_grad=True)
    out = nf4_linear(
        x,
        state["packed"],
        state["quantized_scalers"],
        state["quantization_factor"],
        state["scaler_mean"],
        codebook,
        block=state["block_size"],
        sb=state["scaler_block_size"],
    )
    out.sum().backward()
    assert x.grad is not None
    assert float(x.grad.abs().max()) > 0
    saved = getattr(out.grad_fn, "saved_tensors", ())
    assert all(tuple(t.shape) != (256, 512) for t in saved)


def test_nf4_scratch_path_matches_dequant_gemm():
    state, codebook = _state(512, 1024)
    x = tp.randn((64, 1024), device="cuda", dtype=tp.bfloat16, requires_grad=True)
    kwargs = dict(block=state["block_size"], sb=state["scaler_block_size"])
    args = (
        state["packed"],
        state["quantized_scalers"],
        state["quantization_factor"],
        state["scaler_mean"],
        codebook,
    )
    got = nf4_linear_scratch(x, *args, **kwargs)
    ref = F.linear(x, dequantize_nf4(state, dtype=tp.bfloat16))
    assert float((got - ref).abs().max()) == 0.0
    grad = tp.randn_like(got)
    dx = tp.autograd.grad(got, x, grad)[0]
    dx_ref = tp.autograd.grad(ref, x, grad, retain_graph=True)[0]
    assert float((dx - dx_ref).abs().max()) == 0.0
    # No dense weight is retained: only packed state and the codebook.
    saved = getattr(got.grad_fn, "saved_tensors", ())
    assert all(tuple(t.shape) != (512, 1024) for t in saved)
