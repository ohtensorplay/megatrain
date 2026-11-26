# Copyright 2026 TensorPlay Authors
# SPDX-License-Identifier: Apache-2.0
"""Generic training objectives on TensorPlay.

``sft_loss`` is the next-token cross-entropy used by pretraining and SFT. It
also accepts optional frozen teacher log-probabilities, in which case an
estimator-based KL term is added (on-policy distillation). No rollout data,
rewards, advantages or importance ratios are involved: those belong to the RL
layer.

Loss functions return ``(loss, diagnostics)``. The trainer computes a masked
mean per microbatch and weights microbatches by their share of target tokens,
so accumulation is exact for any token-budget packing.
"""

from __future__ import annotations

import tensorplay as tp
import tensorplay.nn.functional as F

from .batch import TrainBatch
from .kl import approx_kl

LOSS_TYPES = ("sft", "pretrain", "distill")

__all__ = [
    "LOSS_TYPES",
    "TensorBatch",
    "masked_mean",
    "sft_loss",
    "sft_loss_with_head",
    "token_logprobs",
]


class TensorBatch:
    """TensorPlay tensors for one :class:`~megatrain.batch.TrainBatch`."""

    def __init__(self, batch: TrainBatch, device: str = "cpu") -> None:
        self.input_ids = tp.tensor(batch.input_ids, dtype=tp.long, device=device)
        self.attention_mask = tp.tensor(
            batch.attention_mask, dtype=tp.float32, device=device
        )
        self.loss_mask = tp.tensor(batch.loss_mask, dtype=tp.float32, device=device)
        self.cu_seqlens = tp.tensor(batch.cu_seqlens, dtype=tp.long, device=device)
        self.max_len = batch.max_len


def token_logprobs(logits, input_ids):
    """Log-probability of each target token, shape ``[B, L]``.

    Uses ``gather(logits) - logsumexp(logits)`` instead of materializing a full
    ``[B, L, vocab]`` ``log_softmax``. NOTE: on TensorPlay 1.3 / RTX 3090 the
    analogous CE path measured ~17% slower and higher peak memory than the
    native kernel; benchmark this path on your hardware before enabling it as
    a production optimization.
    """
    target = tp.gather(logits, -1, input_ids.unsqueeze(-1)).squeeze(-1)
    return target - tp.logsumexp(logits, dim=-1)


def masked_mean(x, mask, denom=None):
    """Masked mean with an explicit denominator (caller controls normalization)."""
    keep = mask.to(x.dtype)
    denominator = keep.sum().clamp(min=1.0) if denom is None else denom
    return (x * keep).sum() / denominator


def sft_loss(
    logits,
    batch: TensorBatch,
    *,
    teacher_logprobs=None,
    distill_coef: float = 0.0,
    kl_type: str = "k3",
    label_smoothing: float = 0.0,
    ce_chunk_tokens: int | None = None,
    ce_impl: str = "native",
):
    """Next-token CE with optional frozen-teacher KL.

    Shift by one: logits at position ``t`` predict ``input_ids[t+1]``. The
    target mask is therefore ``loss_mask[:, 1:]`` — a 1 marks a target token,
    so prompt positions and padding never contribute.

    ``teacher_logprobs`` must be per-token log-probs of shape ``[B, L]`` from a
    frozen model; ``distill_coef > 0`` adds ``coef * KL(student || teacher)``.

    ``ce_chunk_tokens`` and ``ce_impl="logsumexp"`` are experimental escape
    hatches, NOT production optimizations: on TensorPlay 1.3 / RTX 3090 both
    measured slower (0.83x) with higher peak memory than the native CE. Keep
    the defaults unless your own A/B disagrees. ``logsumexp`` only supports
    ``label_smoothing == 0``.
    """
    if teacher_logprobs is None and distill_coef:
        raise ValueError("distill_coef > 0 requires teacher_logprobs")
    shift_logits = logits[:, :-1, :]
    shift_ids = batch.input_ids[:, 1:]
    shift_mask = batch.loss_mask[:, 1:]
    vocab = shift_logits.shape[-1]
    flat_logits = shift_logits.reshape(-1, vocab)
    flat_ids = shift_ids.reshape(-1)
    if ce_chunk_tokens is not None and ce_chunk_tokens < 1:
        raise ValueError("ce_chunk_tokens must be >= 1 when set")
    if ce_impl not in ("native", "native_ignore", "logsumexp"):
        raise ValueError("ce_impl must be 'native', 'native_ignore' or 'logsumexp'")
    if ce_impl == "logsumexp" and label_smoothing:
        raise ValueError("ce_impl='logsumexp' requires label_smoothing == 0")

    def _native(rows, targets, mask_rows):
        return F.cross_entropy(
            rows, targets, reduction="none", label_smoothing=label_smoothing
        )

    def _native_ignore(rows, targets, mask_rows):
        # CE with ignore_index zeroes masked targets inside the native kernel,
        # removing one elementwise multiply from the common SFT path.
        safe = targets.masked_fill(mask_rows == 0, -100)
        return F.cross_entropy(
            rows, safe, reduction="none", ignore_index=-100,
            label_smoothing=label_smoothing,
        )

    def _logsumexp(rows, targets, mask_rows):
        log_norm = tp.logsumexp(rows, dim=-1)
        target_logit = tp.gather(rows, -1, targets.unsqueeze(-1)).squeeze(-1)
        return log_norm - target_logit

    compute = {
        "native": _native,
        "native_ignore": _native_ignore,
        "logsumexp": _logsumexp,
    }[ce_impl]
    flat_mask = shift_mask.reshape(-1)
    if ce_chunk_tokens is not None and flat_logits.shape[0] > ce_chunk_tokens:
        pieces = [
            compute(
                flat_logits[start : start + ce_chunk_tokens],
                flat_ids[start : start + ce_chunk_tokens],
                flat_mask[start : start + ce_chunk_tokens],
            )
            for start in range(0, flat_logits.shape[0], ce_chunk_tokens)
        ]
        token_ce = tp.cat(pieces).reshape(shift_mask.shape)
    else:
        token_ce = compute(flat_logits, flat_ids, flat_mask).reshape(shift_mask.shape)
    keep = shift_mask.to(token_ce.dtype)
    denom = keep.sum().clamp(min=1.0)
    numerator = (
        token_ce.sum() if ce_impl == "native_ignore" else (token_ce * keep).sum()
    )
    ce = numerator / denom
    loss = ce
    diag = {"ce": ce.detach(), "tokens": keep.sum().detach()}
    if teacher_logprobs is not None and distill_coef:
        student = token_logprobs(logits, batch.input_ids)[:, :-1]
        teacher = teacher_logprobs[:, :-1]
        kl = masked_mean(approx_kl(student, teacher, kl_type=kl_type), keep, denom=denom)
        loss = loss + distill_coef * kl
        diag["kl"] = kl.detach()
    return loss, diag


def sft_loss_with_head(
    hidden,
    batch: TensorBatch,
    head_weight,
    *,
    head_bias=None,
    teacher_logprobs=None,
    distill_coef: float = 0.0,
    kl_type: str = "k3",
    label_smoothing: float = 0.0,
):
    """Next-token CE whose projection never materializes its logits.

    Same objective, mask, denominator and diagnostics as :func:`sft_loss`, but
    the vocabulary projection is fused into the loss through
    ``F.linear_cross_entropy``: the ``[tokens, vocab]`` logits are written once
    by the matmul and are then reused by the loss backward as its own gradient
    buffer. Nothing keeps a second full-size tensor, so a bf16 step pays for
    one logits buffer instead of three (plus the fp32 loss workspace).

    Args:
        hidden: ``[B, L, hidden]`` states feeding the output projection.
        batch: the batch whose ``input_ids`` / ``loss_mask`` define the targets.
        head_weight: ``[vocab, hidden]`` projection weight.  Passed as a keyword
            from a trainer's ``loss_kwargs`` so the positional contract stays
            ``loss_fn(model_output, batch)``.

    Returns:
        ``(loss, diagnostics)``, exactly like :func:`sft_loss`.

    Note:
        Class weights, label smoothing and the distillation term need the
        logits themselves; those configurations delegate to :func:`sft_loss`
        and therefore keep the ordinary path.
    """
    if teacher_logprobs is not None and distill_coef:
        logits = F.linear(hidden, head_weight, head_bias)
        return sft_loss(
            logits, batch, teacher_logprobs=teacher_logprobs,
            distill_coef=distill_coef, kl_type=kl_type,
            label_smoothing=label_smoothing,
        )
    if label_smoothing:
        logits = F.linear(hidden, head_weight, head_bias)
        return sft_loss(logits, batch, label_smoothing=label_smoothing)

    vocab = head_weight.shape[0]
    ids = batch.input_ids[:, 1:].reshape(-1)
    mask = batch.loss_mask[:, 1:].reshape(-1)
    # -100 is outside the vocabulary, so it can only come from this masking.
    targets = ids.masked_fill(mask == 0, -100)
    rows = hidden[:, :-1, :].reshape(-1, hidden.shape[-1])
    token_ce = F.linear_cross_entropy(
        rows, head_weight, targets, linear_bias=head_bias,
        ignore_index=-100, reduction="none",
    )
    keep = mask.to(token_ce.dtype)
    denom = keep.sum().clamp(min=1.0)
    ce = token_ce.sum() / denom
    return ce, {"ce": ce.detach(), "tokens": keep.sum().detach(), "vocab": vocab}
