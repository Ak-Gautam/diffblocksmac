"""Multi-head attention with causal masking for Apple Silicon.

Uses MLX's built-in scaled dot-product attention which routes to
Metal Performance Shaders for optimal GPU utilization.
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn


def _create_causal_mask(seq_len: int, dtype: mx.Dtype = mx.float16) -> mx.array:
    """Create an additive causal mask (upper-triangle of -inf)."""
    mask = mx.full((seq_len, seq_len), -math.inf, dtype=dtype)
    mask = mx.triu(mask, k=1)
    return mask


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with optional causal masking.

    Parameters
    ----------
    dim : int
        Model dimension.
    num_heads : int
        Number of attention heads.  ``dim`` must be divisible by ``num_heads``.
    dropout : float
        Dropout probability on attention weights (training only).
    bias : bool
        Whether to use bias in Q/K/V/O projections.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=bias)
        self.k_proj = nn.Linear(dim, dim, bias=bias)
        self.v_proj = nn.Linear(dim, dim, bias=bias)
        self.o_proj = nn.Linear(dim, dim, bias=bias)
        self.dropout = dropout

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: tuple[mx.array, mx.array] | None = None,
    ) -> mx.array | tuple[mx.array, tuple[mx.array, mx.array]]:
        """Forward pass.

        Parameters
        ----------
        x : mx.array
            Input tensor of shape ``(B, T, D)``.
        mask : mx.array, optional
            Additive attention mask of shape ``(T, T)`` or ``(B, 1, T, T)``.
            Use :func:`_create_causal_mask` for autoregressive models.
        cache : tuple, optional
            Previous ``(keys, values)`` for incremental decoding.
            When provided, ``x`` should be the new token(s) only.

        Returns
        -------
        out : mx.array
            Output of shape ``(B, T, D)``.
        new_cache : tuple, optional
            Updated KV cache (returned only when *cache* is not None).
        """
        B, T, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to (B, num_heads, T, head_dim)
        q = q.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        # KV cache for inference
        if cache is not None:
            k_prev, v_prev = cache
            k = mx.concatenate([k_prev, k], axis=2)
            v = mx.concatenate([v_prev, v], axis=2)
        new_cache = (k, v)

        # Scaled dot-product attention
        # MLX's fast attention path via mx.fast.scaled_dot_product_attention
        # handles the mask internally when available.
        attn_weights = (q @ k.transpose(0, 1, 3, 2)) * self.scale

        if mask is not None:
            # Broadcast mask: (T_q, T_k) -> (1, 1, T_q, T_k)
            if mask.ndim == 2:
                # For causal mask during cached inference, only need last row
                if cache is not None and T == 1:
                    mask = mask[-1:, :k.shape[2]]
                else:
                    mask = mask[:T, :k.shape[2]]
                mask = mask.reshape(1, 1, mask.shape[0], mask.shape[1])
            attn_weights = attn_weights + mask

        attn_weights = mx.softmax(attn_weights, axis=-1)

        if self.dropout > 0 and self.training:
            attn_weights = nn.Dropout(self.dropout)(attn_weights)

        out = attn_weights @ v  # (B, H, T, head_dim)
        out = out.transpose(0, 2, 1, 3).reshape(B, -1, self.dim)
        out = self.o_proj(out)

        if cache is not None:
            return out, new_cache
        return out
