"""Text dataset utilities for autoregressive language model training.

Uses the native C extension for fast batch sampling and sequence
extraction when available (xoshiro256** PRNG + memcpy vs numpy).
Falls back to numpy transparently.
"""

from __future__ import annotations

import numpy as np
import mlx.core as mx

from .._native import (
    extract_sequences,
    is_available as native_available,
    sample_batch_indices,
    seed_rng,
)


class CharDataset:
    """Character-level text dataset.

    Converts a text string into integer-encoded character sequences.
    Useful for quick experiments and sanity-checking the training loop.

    Parameters
    ----------
    text : str
        The raw text to train on.
    seq_len : int
        Sequence length for each training example.

    Example
    -------
    >>> ds = CharDataset("hello world " * 1000, seq_len=64)
    >>> batch = ds.get_batch(batch_size=8)
    >>> batch.shape
    (8, 64)
    """

    def __init__(self, text: str, seq_len: int = 128):
        chars = sorted(set(text))
        self.char_to_idx = {c: i for i, c in enumerate(chars)}
        self.idx_to_char = {i: c for c, i in self.char_to_idx.items()}
        self.vocab_size = len(chars)
        self.seq_len = seq_len

        # Encode the full text as a contiguous numpy array
        self._data = np.array(
            [self.char_to_idx[c] for c in text], dtype=np.int32
        )

    def __len__(self) -> int:
        return max(0, len(self._data) - self.seq_len)

    def get_batch(self, batch_size: int) -> mx.array:
        """Sample a random batch of sequences.

        Uses the native C extension for index sampling and sequence
        extraction when available (2-5x faster for typical batch sizes).

        Returns
        -------
        mx.array
            Token IDs of shape ``(batch_size, seq_len)``.
        """
        max_start = len(self._data) - self.seq_len
        if max_start <= 0:
            raise ValueError("Text is too short for the given seq_len")

        # Fast path: C-accelerated sampling + extraction
        starts = sample_batch_indices(batch_size, max_start)
        batch = extract_sequences(self._data, starts, self.seq_len)

        return mx.array(batch)

    def decode(self, tokens: mx.array | list[int]) -> str:
        """Decode token IDs back to text."""
        if isinstance(tokens, mx.array):
            tokens = tokens.tolist()
        if isinstance(tokens[0], list):
            tokens = tokens[0]
        return "".join(self.idx_to_char.get(t, "?") for t in tokens)


class TextBatcher:
    """Infinite batch iterator over a text dataset.

    Wraps a :class:`CharDataset` (or any object with a ``get_batch``
    method) to produce an infinite stream of batches for training.

    Parameters
    ----------
    dataset : CharDataset
        The dataset to sample from.
    batch_size : int
        Number of sequences per batch.

    Example
    -------
    >>> ds = CharDataset(text, seq_len=64)
    >>> batcher = TextBatcher(ds, batch_size=32)
    >>> for batch in batcher:
    ...     trainer.train_step(batch)
    """

    def __init__(self, dataset: CharDataset, batch_size: int = 32):
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        """Yield batches infinitely."""
        while True:
            yield self.dataset.get_batch(self.batch_size)


def load_text_file(path: str, encoding: str = "utf-8") -> str:
    """Load a text file."""
    with open(path, "r", encoding=encoding) as f:
        return f.read()


def create_dataset(
    text: str,
    *,
    seq_len: int = 128,
    batch_size: int = 32,
) -> tuple[CharDataset, TextBatcher]:
    """Convenience function to create a dataset and batcher.

    Returns
    -------
    tuple
        ``(dataset, batcher)`` where batcher yields infinite batches.
    """
    ds = CharDataset(text, seq_len=seq_len)
    batcher = TextBatcher(ds, batch_size=batch_size)
    return ds, batcher
