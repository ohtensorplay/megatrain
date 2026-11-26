"""Tiny real TensorPlay models on CPU: no stubs, no fake optimizers."""

import pytest
import tensorplay as tp


class TinyPolicy(tp.nn.Module):
    def __init__(self, vocab: int = 8, hidden: int = 8) -> None:
        super().__init__()
        self.emb = tp.nn.Embedding(vocab, hidden)
        self.head = tp.nn.Linear(hidden, vocab)

    def forward(self, input_ids):
        return self.head(self.emb(input_ids))


@pytest.fixture
def policy():
    tp.manual_seed(1234)
    return TinyPolicy()
