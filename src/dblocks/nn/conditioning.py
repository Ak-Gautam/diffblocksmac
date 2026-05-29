"""Noise-level conditioning modules for DiffusionBlocks.

These modules inject sigma information into transformer layers so each
block knows which noise level it is denoising.  Follows the DiT
(Diffusion Transformer) conditioning design.
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn


class TimestepEmbedder(nn.Module):
    """Embed a scalar noise level into a dense vector.

    Uses sinusoidal positional encoding followed by a two-layer MLP,
    matching the DiT / EDM convention where the input is
    ``c_noise = 0.25 * ln(sigma)``.

    Parameters
    ----------
    hidden_size : int
        Output embedding dimension.
    freq_dim : int
        Dimension of the sinusoidal frequency encoding.
    """

    def __init__(self, hidden_size: int, freq_dim: int = 256):
        super().__init__()
        self.freq_dim = freq_dim
        self.linear1 = nn.Linear(freq_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)

    @staticmethod
    def _sinusoidal_embedding(
        t: mx.array, dim: int, max_period: float = 10000.0,
    ) -> mx.array:
        """Compute sinusoidal positional encoding for scalar *t*.

        Parameters
        ----------
        t : mx.array
            Shape ``(B,)`` — the noise-level conditioning values.
        dim : int
            Encoding dimension (must be even).
        """
        half = dim // 2
        freqs = mx.exp(
            -math.log(max_period) * mx.arange(half, dtype=mx.float32) / half
        )
        # t: (B,) -> (B, 1) * (1, half) -> (B, half)
        args = t[:, None].astype(mx.float32) * freqs[None, :]
        emb = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)
        if dim % 2 == 1:
            emb = mx.concatenate([emb, mx.zeros_like(emb[:, :1])], axis=-1)
        return emb

    def __call__(self, t: mx.array) -> mx.array:
        """Embed noise conditioning.

        Parameters
        ----------
        t : mx.array
            Shape ``(B,)`` — typically ``c_noise = 0.25 * ln(sigma)``.

        Returns
        -------
        mx.array
            Shape ``(B, hidden_size)``.
        """
        t_freq = self._sinusoidal_embedding(t, self.freq_dim)
        t_freq = t_freq.astype(self.linear1.weight.dtype)
        return self.linear2(nn.silu(self.linear1(t_freq)))


class AdaLN(nn.Module):
    """Adaptive Layer Normalization with gating.

    Produces *6* modulation parameters from the conditioning vector:
    ``(shift_attn, scale_attn, gate_attn, shift_ffn, scale_ffn, gate_ffn)``.

    The gate outputs are zero-initialized for stable training start
    (AdaLN-Zero from DiT).

    Parameters
    ----------
    hidden_size : int
        Model hidden dimension (= conditioning vector dimension).
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, 6 * hidden_size)
        # Zero-initialize the linear layer for AdaLN-Zero
        self.linear.weight = mx.zeros_like(self.linear.weight)
        self.linear.bias = mx.zeros_like(self.linear.bias)

    def __call__(self, c: mx.array) -> tuple[mx.array, ...]:
        """Compute modulation parameters.

        Parameters
        ----------
        c : mx.array
            Conditioning vector of shape ``(B, hidden_size)``.

        Returns
        -------
        tuple of 6 mx.array
            Each of shape ``(B, hidden_size)``:
            ``(shift_attn, scale_attn, gate_attn,
              shift_ffn, scale_ffn, gate_ffn)``.
        """
        out = nn.silu(c)
        out = self.linear(out)
        # Split into 6 equal chunks
        chunks = mx.split(out, 6, axis=-1)
        return tuple(chunks)


def modulate(x: mx.array, shift: mx.array, scale: mx.array) -> mx.array:
    """Apply AdaLN modulation: ``x * (1 + scale) + shift``.

    Parameters
    ----------
    x : mx.array
        Input of shape ``(B, T, D)``.
    shift, scale : mx.array
        Modulation params of shape ``(B, D)`` — broadcast over *T*.
    """
    return x * (1.0 + scale[:, None, :]) + shift[:, None, :]
