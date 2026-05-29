"""Feed-forward network variants for transformer blocks."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class FeedForward(nn.Module):
    """Standard two-layer MLP with GELU activation.

    Parameters
    ----------
    dim : int
        Input / output dimension.
    mult : float
        Hidden dimension multiplier.  Hidden dim = ``int(dim * mult)``.
    dropout : float
        Dropout probability after the activation.
    bias : bool
        Whether to use bias in linear layers.
    """

    def __init__(
        self,
        dim: int,
        mult: float = 4.0,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        hidden = int(dim * mult)
        self.w1 = nn.Linear(dim, hidden, bias=bias)
        self.w2 = nn.Linear(hidden, dim, bias=bias)
        self.dropout = dropout

    def __call__(self, x: mx.array) -> mx.array:
        h = nn.gelu_approx(self.w1(x))
        if self.dropout > 0 and self.training:
            h = nn.Dropout(self.dropout)(h)
        return self.w2(h)


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network.

    Uses the gated linear unit with SiLU activation, which gives better
    quality for language models.  The effective hidden dim is
    ``int(2/3 * dim * mult)`` to match parameter count with standard MLP.

    Parameters
    ----------
    dim : int
        Input / output dimension.
    mult : float
        Hidden dimension multiplier.
    bias : bool
        Whether to use bias in linear layers.
    """

    def __init__(
        self,
        dim: int,
        mult: float = 4.0,
        bias: bool = False,
    ):
        super().__init__()
        # 2/3 factor so total params ≈ standard MLP with same mult
        hidden = int(2 * dim * mult / 3)
        # Round to nearest multiple of 64 for Metal alignment
        hidden = ((hidden + 63) // 64) * 64

        self.w_gate = nn.Linear(dim, hidden, bias=bias)
        self.w_up = nn.Linear(dim, hidden, bias=bias)
        self.w_down = nn.Linear(hidden, dim, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        return self.w_down(nn.silu(self.w_gate(x)) * self.w_up(x))
