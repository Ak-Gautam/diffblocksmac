"""DiffusionBlocks-aware transformer layers.

Each layer receives noise-level conditioning via AdaLN, enabling
independent block-wise training with the denoising objective.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from .attention import MultiHeadAttention
from .conditioning import AdaLN, modulate
from .feedforward import FeedForward, SwiGLU


class DBlockTransformerLayer(nn.Module):
    """Single transformer layer with AdaLN-Zero conditioning.

    Architecture:
        1. AdaLN-modulated self-attention with residual
        2. AdaLN-modulated feed-forward with residual

    Parameters
    ----------
    dim : int
        Model dimension.
    num_heads : int
        Number of attention heads.
    mlp_ratio : float
        Feed-forward hidden dimension multiplier.
    dropout : float
        Dropout probability.
    use_swiglu : bool
        If True, use SwiGLU instead of standard GELU MLP.
    bias : bool
        Whether to use bias in linear layers.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        use_swiglu: bool = True,
        bias: bool = False,
    ):
        super().__init__()
        self.attn_norm = nn.RMSNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads, dropout=dropout, bias=bias)
        self.ffn_norm = nn.RMSNorm(dim)
        if use_swiglu:
            self.ffn = SwiGLU(dim, mult=mlp_ratio, bias=bias)
        else:
            self.ffn = FeedForward(dim, mult=mlp_ratio, dropout=dropout, bias=bias)
        self.adaln = AdaLN(dim)

    def __call__(
        self,
        x: mx.array,
        cond: mx.array,
        mask: mx.array | None = None,
        cache: tuple[mx.array, mx.array] | None = None,
    ) -> mx.array | tuple[mx.array, tuple[mx.array, mx.array]]:
        """Forward pass.

        Parameters
        ----------
        x : mx.array
            Input tensor ``(B, T, D)``.
        cond : mx.array
            Conditioning vector ``(B, D)`` from :class:`TimestepEmbedder`.
        mask : mx.array, optional
            Causal attention mask.
        cache : tuple, optional
            KV cache for incremental decoding.
        """
        # Get all 6 modulation params
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = self.adaln(cond)

        # Self-attention with AdaLN
        h = modulate(self.attn_norm(x), shift_a, scale_a)
        attn_out = self.attn(h, mask=mask, cache=cache)
        if cache is not None:
            attn_out, new_cache = attn_out
        x = x + gate_a[:, None, :] * attn_out

        # Feed-forward with AdaLN
        h = modulate(self.ffn_norm(x), shift_f, scale_f)
        x = x + gate_f[:, None, :] * self.ffn(h)

        if cache is not None:
            return x, new_cache
        return x


class TransformerBlock(nn.Module):
    """A group of transformer layers forming one DiffusionBlocks block.

    This is the unit of independent training: during a training step,
    exactly one ``TransformerBlock`` is unfrozen and trained to denoise.

    Parameters
    ----------
    dim : int
        Model dimension.
    num_heads : int
        Number of attention heads per layer.
    num_layers : int
        Number of transformer layers in this block.
    mlp_ratio : float
        Feed-forward expansion ratio.
    dropout : float
        Dropout probability.
    use_swiglu : bool
        Whether to use SwiGLU feed-forward.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_layers: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        use_swiglu: bool = True,
    ):
        super().__init__()
        self.layers = [
            DBlockTransformerLayer(
                dim, num_heads, mlp_ratio=mlp_ratio,
                dropout=dropout, use_swiglu=use_swiglu,
            )
            for _ in range(num_layers)
        ]

    def __call__(
        self,
        x: mx.array,
        cond: mx.array,
        mask: mx.array | None = None,
        cache: list[tuple[mx.array, mx.array]] | None = None,
    ) -> mx.array | tuple[mx.array, list[tuple[mx.array, mx.array]]]:
        """Run all layers in this block sequentially.

        Parameters
        ----------
        x : mx.array
            Input ``(B, T, D)``.
        cond : mx.array
            Conditioning ``(B, D)``.
        mask : mx.array, optional
            Causal mask.
        cache : list, optional
            Per-layer KV caches for inference.
        """
        new_caches: list[tuple[mx.array, mx.array]] = []

        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache is not None else None
            out = layer(x, cond, mask=mask, cache=layer_cache)
            if cache is not None:
                x, kv = out
                new_caches.append(kv)
            else:
                x = out

        if cache is not None:
            return x, new_caches
        return x
