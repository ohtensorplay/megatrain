# Copyright 2026 TensorPlay Authors
# SPDX-License-Identifier: Apache-2.0
"""The fused vocabulary projection + CE path.

Two things must hold before this can be called a production optimization:

1. arithmetic equivalence -- same loss, same gradients as the reference the
   stack already had, at a precision the comparison with Megatron accepts;
2. the saving it claims is real -- fewer full-size buffers alive during the
   step, measured rather than asserted.

Both are checked here, on the GPU when one is present (the path is CUDA-only and
falls back to the composed reference elsewhere).

The kernel-dependent checks skip when the installed TensorPlay does not carry
the local patch described in ``devdocs/performance.md`` §4.6: without it,
``F.linear_cross_entropy`` composes the reference and proves nothing about the
kernels.  The objective-level checks run either way.
"""

from __future__ import annotations

import importlib.util

import pytest
import tensorplay as tp
import tensorplay.nn.functional as F

from megatrain.losses import TensorBatch, sft_loss, sft_loss_with_head

CUDA = tp.cuda.is_available()
cuda_only = pytest.mark.skipif(not CUDA, reason="fused path needs CUDA")


def _module_exists(name: str) -> bool:
    """Whether an importable module exists, without importing it."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError):
        return False


#: Upstream TensorPlay serves ``linear_cross_entropy`` as a reference
#: composition; only the patched tree has the kernels these tests measure.
HAS_FUSED_KERNEL = _module_exists("tensorplay.nn._fused_ce")
requires_fused = pytest.mark.skipif(
    not HAS_FUSED_KERNEL,
    reason="needs the local TensorPlay fused-CE patch (see devdocs/performance.md)",
)

TOKENS, HIDDEN, VOCAB = 256, 64, 4096


def _head_inputs(seed: int = 0, dtype=tp.float32):
    tp.manual_seed(seed)
    x = tp.randn(TOKENS, HIDDEN, device="cuda", dtype=dtype).requires_grad_(True)
    w = tp.randn(VOCAB, HIDDEN, device="cuda", dtype=dtype) * 0.02
    w.requires_grad_(True)
    ids = tp.randint(0, VOCAB, (TOKENS,), device="cuda")
    return x, w, ids


@requires_fused
@cuda_only
def test_fused_linear_ce_matches_reference_loss_and_grads():
    x, w, ids = _head_inputs()
    want = F.cross_entropy(F.linear(x, w), ids)
    want.backward()
    want_loss = float(want.detach())
    want_dx = x.grad.float().clone()
    want_dw = w.grad.float().clone()

    x.grad = None
    w.grad = None
    got = F.linear_cross_entropy(x, w, ids)
    got.backward()
    loss = float(got.detach())

    assert abs(loss - want_loss) <= 1e-6 * abs(want_loss)
    for name, got_grad, want_grad in (
        ("dinput", x.grad.float(), want_dx),
        ("dweight", w.grad.float(), want_dw),
    ):
        scale = float(want_grad.abs().max())
        assert float((got_grad - want_grad).abs().max()) <= 1e-5 * scale + 1e-8, name


@requires_fused
@cuda_only
def test_fused_linear_ce_ignores_masked_rows():
    x, w, ids = _head_inputs(1)
    masked = ids.clone()
    masked[: TOKENS // 4] = -100

    want = F.cross_entropy(F.linear(x, w), masked, ignore_index=-100)
    want.backward()
    want_loss = float(want.detach())
    want_dw = w.grad.float().clone()

    w.grad = None
    got = F.linear_cross_entropy(x, w, masked, ignore_index=-100)
    got.backward()
    assert abs(float(got.detach()) - want_loss) <= 1e-6 * abs(want_loss)
    scale = float(want_dw.abs().max())
    assert float((w.grad.float() - want_dw).abs().max()) <= 1e-5 * scale + 1e-8


@requires_fused
@cuda_only
def test_fused_path_keeps_one_logits_buffer():
    """The claim: the loss does not add full-size buffers to the step."""
    rows, vocab = 2048, 65536
    tp.manual_seed(0)
    logits = (tp.randn(rows, vocab, device="cuda") * 0.5).to(tp.bfloat16)
    ids = tp.randint(0, vocab, (rows,), device="cuda")

    def peak(fn):
        tp.cuda.synchronize()
        tp.cuda.empty_cache()
        tp.cuda.reset_peak_memory_stats()
        fn()
        tp.cuda.synchronize()
        value = tp.cuda.max_memory_allocated() / 1024**2
        return value

    def reference():
        buf = logits.clone().requires_grad_(True)
        F.cross_entropy(buf, ids).backward()

    def fused():
        buf = logits.clone().requires_grad_(True)
        from tensorplay.nn._fused_ce import fused_cross_entropy

        fused_cross_entropy(buf, ids).backward()

    reference()
    fused()
    ref_peak, fused_peak = peak(reference), peak(fused)
    # The reference holds an fp32 upcast of the logits for the softmax on top of
    # its own output; the fused walk keeps the logits and two per-row vectors,
    # so at least one fp32 logits buffer (56 MiB here) must disappear.
    fp32_logits_mib = rows * vocab * 4 / 1024**2
    assert ref_peak - fused_peak >= 0.85 * fp32_logits_mib, (fused_peak, ref_peak)


def test_head_loss_matches_logits_loss():
    """``sft_loss_with_head`` is the same objective as ``sft_loss``."""

    from megatrain import TrainSample, collate

    batch = collate([TrainSample([1, 2, 3, 4, 5])])
    tensors = TensorBatch(batch, device="cpu")
    hidden = tp.randn(1, 5, 8, device="cpu")
    head = tp.randn(16, 8, device="cpu") * 0.1

    logits = F.linear(hidden, head)
    want, want_diag = sft_loss(logits, tensors)
    got, got_diag = sft_loss_with_head(hidden, tensors, head)
    assert abs(float(got.detach()) - float(want.detach())) <= 1e-6
    assert float(got_diag["tokens"]) == float(want_diag["tokens"])


def test_head_loss_falls_back_for_label_smoothing():
    from megatrain import TrainSample, collate

    batch = collate([TrainSample([1, 2, 3, 4])])
    tensors = TensorBatch(batch, device="cpu")
    hidden = tp.randn(1, 4, 8, device="cpu")
    head = tp.randn(16, 8, device="cpu") * 0.1
    got, _ = sft_loss_with_head(hidden, tensors, head, label_smoothing=0.1)
    want, _ = sft_loss(F.linear(hidden, head), tensors, label_smoothing=0.1)
    assert abs(float(got.detach()) - float(want.detach())) <= 1e-6


class _HiddenHead(tp.nn.Module):
    """Returns hidden states; the head lives beside them, not on top."""

    def __init__(self, vocab: int = 32, hidden: int = 16) -> None:
        super().__init__()
        self.emb = tp.nn.Embedding(vocab, hidden)
        self.head = tp.nn.Linear(hidden, vocab, bias=False)

    def forward(self, input_ids):
        return self.emb(input_ids)


@cuda_only
def test_trainer_step_with_head_loss_updates_head_and_embedding():
    from megatrain import BaseTrainer, TrainerConfig, TrainSample, collate

    tp.manual_seed(7)
    model = _HiddenHead().to("cuda")
    trainer = BaseTrainer(model, TrainerConfig(device="cuda", learning_rate=1e-2))
    batch = collate(
        [TrainSample([1, 2, 3, 4, 5], [0, 0, 1, 1, 1]), TrainSample([6, 7, 8], [0, 1, 1])]
    )
    before_head = trainer.policy.head.weight.detach().clone()
    before_emb = trainer.policy.emb.weight.detach().clone()

    metrics = trainer.train_step(
        batch,
        loss_fn=sft_loss_with_head,
        loss_kwargs={"head_weight": trainer.policy.head.weight},
    )

    assert metrics["train/loss"] > 0
    assert abs(float(tp.tensor(metrics["train/loss"]))) < 1e3
    moved_head = float((trainer.policy.head.weight.detach() - before_head).abs().sum())
    moved_emb = float((trainer.policy.emb.weight.detach() - before_emb).abs().sum())
    assert moved_head > 0, "the head never left the loss's own buffer"
    assert moved_emb > 0, "gradient did not reach the trunk"
    assert trainer.step_count == 1


def test_module_probe_distinguishes_present_from_absent():
    assert _module_exists("tensorplay.nn")
    assert not _module_exists("tensorplay.nn._definitely_not_a_module")
