# Copyright 2026 TensorPlay Authors
# SPDX-License-Identifier: Apache-2.0
"""Backend-agnostic minibatches.

A :class:`TrainBatch` carries only what a forward/backward step needs: token
ids, masks, positions and varlen boundaries. There are no policy versions,
rewards or behavior logprobs here; an RL layer can subclass or convert its own
records into this shape without leaking the wire protocol into the trainer.
"""

from __future__ import annotations

import dataclasses
import random
from typing import Sequence


@dataclasses.dataclass(frozen=True, slots=True)
class TrainSample:
    """One tokenized training example.

    ``loss_mask`` marks **target** tokens (a 1 at position ``t`` means the
    token at ``t`` contributes as the label of a next-token prediction from
    position ``t-1``). Prompt tokens and padding therefore carry 0.
    """

    token_ids: list[int]
    loss_mask: list[int] | None = None

    @property
    def length(self) -> int:
        return len(self.token_ids)

    def validate(self) -> None:
        if not self.token_ids:
            raise ValueError("sample needs at least one token")
        if self.loss_mask is not None and len(self.loss_mask) != self.length:
            raise ValueError("loss_mask must align with token_ids")
        if self.loss_mask is not None and any(mask not in (0, 1) for mask in self.loss_mask):
            raise ValueError("loss_mask must be binary")


@dataclasses.dataclass(frozen=True, slots=True)
class TrainBatch:
    input_ids: list[list[int]]
    attention_mask: list[list[int]]
    loss_mask: list[list[int]]
    position_ids: list[list[int]]
    cu_seqlens: list[int]
    max_len: int

    @property
    def token_count(self) -> int:
        """Real (non-padding) tokens; ``cu_seqlens[-1]`` is their exact count."""
        return self.cu_seqlens[-1]

    @property
    def size(self) -> int:
        return len(self.input_ids)

    def keep_tokens(self) -> int:
        """Target tokens counted by the loss (position 0 has no predecessor)."""
        return sum(sum(row[1:]) for row in self.loss_mask)


def collate(samples: Sequence[TrainSample]) -> TrainBatch:
    """Right-pad samples and build varlen boundaries for packed attention."""
    if not samples:
        raise ValueError("cannot collate an empty batch")
    for sample in samples:
        sample.validate()
    max_len = max(sample.length for sample in samples)
    input_ids: list[list[int]] = []
    attention_mask: list[list[int]] = []
    loss_mask: list[list[int]] = []
    position_ids: list[list[int]] = []
    cu_seqlens = [0]
    for sample in samples:
        pad = max_len - sample.length
        mask = sample.loss_mask if sample.loss_mask is not None else [1] * sample.length
        input_ids.append(list(sample.token_ids) + [0] * pad)
        attention_mask.append([1] * sample.length + [0] * pad)
        loss_mask.append(list(mask) + [0] * pad)
        position_ids.append(list(range(sample.length)) + [0] * pad)
        cu_seqlens.append(cu_seqlens[-1] + sample.length)
    return TrainBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        loss_mask=loss_mask,
        position_ids=position_ids,
        cu_seqlens=cu_seqlens,
        max_len=max_len,
    )


def slice_batch(batch: TrainBatch, start: int, stop: int) -> TrainBatch:
    """Slice rows of a batch for gradient accumulation."""
    return TrainBatch(
        input_ids=batch.input_ids[start:stop],
        attention_mask=batch.attention_mask[start:stop],
        loss_mask=batch.loss_mask[start:stop],
        position_ids=batch.position_ids[start:stop],
        cu_seqlens=[0]
        + [n - batch.cu_seqlens[start] for n in batch.cu_seqlens[start + 1 : stop + 1]],
        max_len=batch.max_len,
    )


def _chunk(sample: TrainSample, max_tokens: int, alignment: int) -> list[TrainSample]:
    step = max_tokens - (max_tokens % alignment) if alignment > 1 else max_tokens
    if step < 1:
        raise ValueError("max_tokens must be >= alignment")
    if sample.length <= step:
        return [sample]
    mask = sample.loss_mask if sample.loss_mask is not None else [1] * sample.length
    return [
        TrainSample(sample.token_ids[start : start + step], mask[start : start + step])
        for start in range(0, sample.length, step)
    ]


def pack_samples(
    samples: Sequence[TrainSample],
    max_tokens: int,
    *,
    alignment: int = 1,
    shuffle: bool = True,
    seed: int | None = None,
) -> list[TrainBatch]:
    """Token-budget packing for pretraining/SFT.

    Long documents are split into aligned contiguous windows; the windows are
    then greedily bin-packed by length so a batch holds at most ``max_tokens``
    real tokens. ``alignment`` should be ``2 * CP * TP`` for CP/TP training,
    matching the upstream RL packing convention.
    """
    if max_tokens < 1:
        raise ValueError("max_tokens must be >= 1")
    chunks: list[TrainSample] = []
    for sample in samples:
        sample.validate()
        chunks.extend(_chunk(sample, max_tokens, alignment))
    if shuffle:
        random.Random(seed).shuffle(chunks)
    # Longest-first packing keeps padding low and makes batch sizes predictable.
    chunks.sort(key=lambda sample: sample.length, reverse=True)
    groups: list[list[TrainSample]] = []
    current: list[TrainSample] = []
    used = 0
    for chunk in chunks:
        if current and used + chunk.length > max_tokens:
            groups.append(current)
            current, used = [], 0
        current.append(chunk)
        used += chunk.length
    if current:
        groups.append(current)
    return [collate(group) for group in groups]


__all__ = ["TrainBatch", "TrainSample", "collate", "pack_samples", "slice_batch"]
