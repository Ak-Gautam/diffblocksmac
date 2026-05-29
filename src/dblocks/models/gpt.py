"""DiffusionBlocks GPT — autoregressive language model.

This model partitions a GPT-style transformer into independently
trainable blocks.  Each block is conditioned on a noise level (sigma)
via AdaLN-Zero and trained with a denoising objective.  At inference
time the blocks compose sequentially to produce clean representations
for next-token prediction.

Memory advantage
----------------
End-to-end training of a D-layer GPT stores activations for all D layers.
DiffusionBlocks with B blocks stores activations for only D/B layers
(one block) plus the shared embedding/head, reducing peak memory by ~1/B.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import mlx.core as mx
import mlx.nn as nn

from ..core.blocks import BlockSpec, make_block_specs
from ..core.schedules import LogNormalNoise
from ..nn.attention import _create_causal_mask
from ..nn.conditioning import TimestepEmbedder
from ..nn.transformer import TransformerBlock


@dataclass
class GPTConfig:
    """Configuration for :class:`DBlockGPT`.

    Parameters
    ----------
    vocab_size : int
        Vocabulary size.
    dim : int
        Model hidden dimension.
    depth : int
        Total number of transformer layers.
    num_heads : int
        Number of attention heads per layer.
    max_seq_len : int
        Maximum sequence length.
    num_blocks : int
        Number of DiffusionBlocks blocks.
    mlp_ratio : float
        Feed-forward expansion ratio.
    dropout : float
        Dropout probability.
    use_swiglu : bool
        Use SwiGLU feed-forward (recommended for LMs).
    overlap : float
        Sigma interval overlap fraction.
    sigma_data : float
        EDM sigma_data parameter.
    noise : LogNormalNoise
        Noise distribution for block boundaries.
    dtype : mx.Dtype
        Parameter dtype (float16 by default on M-series).
    """
    vocab_size: int = 50257
    dim: int = 768
    depth: int = 12
    num_heads: int = 12
    max_seq_len: int = 1024
    num_blocks: int = 3
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    use_swiglu: bool = True
    overlap: float = 0.1
    sigma_data: float = 0.5
    noise: LogNormalNoise = field(default_factory=LogNormalNoise)
    dtype: mx.Dtype = mx.float16

    # -- Presets -----------------------------------------------------------

    @classmethod
    def small(cls, vocab_size: int = 50257, num_blocks: int = 3) -> GPTConfig:
        """GPT-2 Small (~125M params)."""
        return cls(
            vocab_size=vocab_size, dim=768, depth=12,
            num_heads=12, num_blocks=num_blocks,
        )

    @classmethod
    def medium(cls, vocab_size: int = 50257, num_blocks: int = 4) -> GPTConfig:
        """GPT-2 Medium (~350M params)."""
        return cls(
            vocab_size=vocab_size, dim=1024, depth=24,
            num_heads=16, num_blocks=num_blocks,
        )

    @classmethod
    def large(cls, vocab_size: int = 50257, num_blocks: int = 6) -> GPTConfig:
        """GPT-2 Large (~774M params)."""
        return cls(
            vocab_size=vocab_size, dim=1280, depth=36,
            num_heads=20, num_blocks=num_blocks,
        )

    @classmethod
    def tiny(cls, vocab_size: int = 50257, num_blocks: int = 2) -> GPTConfig:
        """Tiny model for testing (~10M params)."""
        return cls(
            vocab_size=vocab_size, dim=256, depth=6,
            num_heads=4, num_blocks=num_blocks, max_seq_len=512,
        )


class DBlockGPT(nn.Module):
    """GPT language model trained with DiffusionBlocks.

    The model is composed of:
    - Token + position embeddings (shared, always trainable)
    - B transformer blocks, each containing depth/B layers
    - Timestep embedder for sigma conditioning (shared)
    - Final RMSNorm + LM head (shared, always trainable)

    During block-wise training, only one block is unfrozen at a time.
    The shared components (embedding, head, timestep embedder) are always
    unfrozen so they co-adapt with all blocks.

    Parameters
    ----------
    config : GPTConfig
        Model configuration.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # -- Shared components (always trainable) --------------------------
        self.wte = nn.Embedding(config.vocab_size, config.dim)
        self.wpe = nn.Embedding(config.max_seq_len, config.dim)
        self.t_embed = TimestepEmbedder(config.dim)
        self.ln_f = nn.RMSNorm(config.dim)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)

        # Weight tying: lm_head shares weights with wte
        self.lm_head.weight = self.wte.weight

        # -- Transformer blocks (independently trainable) ------------------
        block_specs = make_block_specs(
            config.depth, config.num_blocks,
            noise=config.noise, overlap=config.overlap,
        )
        self._block_specs = block_specs

        layers_per_block = [len(spec.layers) for spec in block_specs]
        self.blocks = [
            TransformerBlock(
                dim=config.dim,
                num_heads=config.num_heads,
                num_layers=n,
                mlp_ratio=config.mlp_ratio,
                dropout=config.dropout,
                use_swiglu=config.use_swiglu,
            )
            for n in layers_per_block
        ]

        # Embed dropout
        self._embed_dropout = config.dropout

        # Apply dtype
        if config.dtype != mx.float32:
            self._apply_dtype(config.dtype)

    def _apply_dtype(self, dtype: mx.Dtype) -> None:
        """Convert all parameters to the target dtype."""
        def _convert(params):
            if isinstance(params, dict):
                return {k: _convert(v) for k, v in params.items()}
            elif isinstance(params, list):
                return [_convert(v) for v in params]
            elif isinstance(params, mx.array) and params.dtype != dtype:
                return params.astype(dtype)
            return params
        self.update(_convert(self.parameters()))

    @property
    def block_specs(self) -> tuple[BlockSpec, ...]:
        """Block specifications in model order."""
        return self._block_specs

    @property
    def num_blocks(self) -> int:
        return len(self.blocks)

    # -- Embedding ---------------------------------------------------------

    def embed(self, tokens: mx.array) -> mx.array:
        """Compute token + position embeddings.

        Parameters
        ----------
        tokens : mx.array
            Token IDs of shape ``(B, T)``.

        Returns
        -------
        mx.array
            Embeddings of shape ``(B, T, D)``.
        """
        B, T = tokens.shape
        pos = mx.arange(T)
        x = self.wte(tokens) + self.wpe(pos)
        if self._embed_dropout > 0 and self.training:
            x = nn.Dropout(self._embed_dropout)(x)
        return x

    # -- Block-level forward -----------------------------------------------

    def forward_block(
        self,
        x: mx.array,
        block_idx: int,
        cond: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        """Run a single transformer block.

        Parameters
        ----------
        x : mx.array
            Input ``(B, T, D)`` — should be pre-scaled by ``c_in``.
        block_idx : int
            Which block to run.
        cond : mx.array
            Conditioning vector ``(B, D)`` from timestep embedder.
        mask : mx.array, optional
            Causal attention mask.
        """
        return self.blocks[block_idx](x, cond, mask=mask)

    # -- Full forward (inference) ------------------------------------------

    def __call__(
        self,
        tokens: mx.array,
        cache: list[list[tuple[mx.array, mx.array]]] | None = None,
    ) -> mx.array | tuple[mx.array, list[list[tuple[mx.array, mx.array]]]]:
        """Full forward pass through all blocks for inference.

        Each block runs at its representative sigma level.  The
        preconditioning (c_skip, c_out, c_in) shapes each block's
        contribution as an Euler denoising step.

        Parameters
        ----------
        tokens : mx.array
            Token IDs ``(B, T)``.
        cache : list, optional
            Nested KV caches ``[block_idx][layer_idx]`` for generation.

        Returns
        -------
        logits : mx.array
            Shape ``(B, T, vocab_size)``.
        new_cache : list, optional
            Updated caches (returned only when ``cache`` is not None).
        """
        x = self.embed(tokens)
        T = tokens.shape[1]

        # Create causal mask
        if cache is not None:
            # During generation, T=1 and we need mask over full seq length
            prev_len = cache[0][0][0].shape[2] if cache[0] else 0
            full_len = prev_len + T
            mask = _create_causal_mask(full_len, dtype=x.dtype)
        else:
            mask = _create_causal_mask(T, dtype=x.dtype)

        new_caches = []
        sd = self.config.sigma_data

        for b_idx in range(self.num_blocks):
            spec = self._block_specs[b_idx]
            sigma = spec.sigma_mid

            # EDM preconditioning
            s2 = sigma ** 2
            d2 = sd ** 2
            denom = (s2 + d2) ** 0.5
            c_skip = d2 / (s2 + d2)
            c_out = sigma * sd / denom
            c_in = 1.0 / denom
            c_noise = 0.25 * math.log(sigma)

            # Conditioning
            c_noise_arr = mx.full((x.shape[0],), c_noise, dtype=x.dtype)
            cond = self.t_embed(c_noise_arr)

            # Forward through block with preconditioning
            block_cache = cache[b_idx] if cache is not None else None
            x_in = c_in * x

            out = self.blocks[b_idx](x_in, cond, mask=mask, cache=block_cache)
            if cache is not None:
                x_pred, bc = out
                new_caches.append(bc)
            else:
                x_pred = out

            x = c_skip * x + c_out * x_pred

        logits = self.lm_head(self.ln_f(x))

        if cache is not None:
            return logits, new_caches
        return logits

    # -- Freeze / unfreeze helpers -----------------------------------------

    def freeze_for_block(self, block_idx: int) -> None:
        """Freeze everything except the given block and shared components.

        Call this before each block-wise training step to ensure only
        the active block (and shared embedding/head) receive gradients.
        """
        self.freeze()
        # Unfreeze shared components
        self.wte.unfreeze()
        self.wpe.unfreeze()
        self.t_embed.unfreeze()
        self.ln_f.unfreeze()
        self.lm_head.unfreeze()
        # Unfreeze the active block
        self.blocks[block_idx].unfreeze()

    # -- Text generation ---------------------------------------------------

    def generate(
        self,
        prompt: mx.array,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> mx.array:
        """Autoregressive text generation.

        Simple implementation without KV cache — recomputes the full
        sequence each step.  Suitable for short generations.

        Parameters
        ----------
        prompt : mx.array
            Token IDs of shape ``(1, T_prompt)``.
        max_new_tokens : int
            Number of tokens to generate.
        temperature : float
            Sampling temperature.
        top_p : float
            Nucleus sampling threshold.

        Returns
        -------
        mx.array
            Generated token IDs ``(1, T_prompt + max_new_tokens)``.
        """
        tokens = prompt

        for _ in range(max_new_tokens):
            # Truncate to max_seq_len if needed
            input_tokens = tokens[:, -self.config.max_seq_len:]

            # Full forward pass (no cache)
            logits = self(input_tokens)
            mx.eval(logits)

            # Get logits for last position
            last_logits = logits[:, -1, :].astype(mx.float32)

            if temperature > 0:
                last_logits = last_logits / temperature

                # Top-p (nucleus) sampling
                if top_p < 1.0:
                    sorted_indices = mx.argsort(last_logits, axis=-1)[:, ::-1]
                    sorted_logits = mx.take_along_axis(last_logits, sorted_indices, axis=-1)
                    sorted_probs = mx.softmax(sorted_logits, axis=-1)
                    cumulative_probs = mx.cumsum(sorted_probs, axis=-1)
                    # Mask out tokens beyond top-p
                    mask = cumulative_probs - sorted_probs > top_p
                    sorted_logits = mx.where(mask, mx.array(-1e9), sorted_logits)
                    last_logits = mx.zeros_like(last_logits)
                    last_logits = last_logits.at[mx.arange(last_logits.shape[0])[:, None], sorted_indices].add(sorted_logits)

                probs = mx.softmax(last_logits, axis=-1)
                next_token = mx.random.categorical(mx.log(probs + 1e-10))
            else:
                next_token = mx.argmax(last_logits, axis=-1)

            next_token = next_token.reshape(1, 1)
            tokens = mx.concatenate([tokens, next_token], axis=1)

        return tokens

    # -- Parameter counting -----------------------------------------------

    def num_parameters(self, trainable_only: bool = False) -> int:
        """Count model parameters."""
        if trainable_only:
            params = self.trainable_parameters()
        else:
            params = self.parameters()

        def _count(p):
            if isinstance(p, mx.array):
                return p.size
            elif isinstance(p, dict):
                return sum(_count(v) for v in p.values())
            elif isinstance(p, list):
                return sum(_count(v) for v in p)
            return 0

        return _count(params)

    def memory_estimate_mb(self) -> dict[str, float]:
        """Estimate memory usage in MB.

        Returns a dict with estimates for:
        - ``params``: total parameter memory
        - ``block_params``: largest single block parameter memory
        - ``shared_params``: shared component parameter memory
        - ``optimizer_full``: Adam state for all params
        - ``optimizer_block``: Adam state for one block + shared
        """
        bytes_per_param = 2 if self.config.dtype == mx.float16 else 4
        mb = 1024 * 1024

        total = self.num_parameters() * bytes_per_param / mb

        # Count per-block params
        block_params = []
        for block in self.blocks:
            def _count(p):
                if isinstance(p, mx.array):
                    return p.size
                elif isinstance(p, dict):
                    return sum(_count(v) for v in p.values())
                elif isinstance(p, list):
                    return sum(_count(v) for v in p)
                return 0
            block_params.append(_count(block.parameters()) * bytes_per_param / mb)

        shared = total - sum(block_params)
        max_block = max(block_params)

        return {
            "params": total,
            "block_params_max": max_block,
            "shared_params": shared,
            "optimizer_full": total * 2,  # Adam: m + v
            "optimizer_block": (max_block + shared) * 2,
            "e2e_peak_estimate": total * 4,  # params + grads + adam m + adam v
            "dblock_peak_estimate": total + (max_block + shared) * 3,  # params + block (grads + m + v)
        }
