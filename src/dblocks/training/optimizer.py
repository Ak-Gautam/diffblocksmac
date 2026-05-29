"""Block-aware optimizer for DiffusionBlocks training.

Uses separate AdamW optimizer instances per block so that optimizer
state (momentum, variance) is block-local.  Shared parameters
(embeddings, LM head) use a dedicated optimizer that is updated
every step.
"""

from __future__ import annotations

import math
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim


def _tree_map(fn, tree):
    """Apply fn to all arrays in a nested dict/list tree."""
    if isinstance(tree, mx.array):
        return fn(tree)
    elif isinstance(tree, dict):
        return {k: _tree_map(fn, v) for k, v in tree.items()}
    elif isinstance(tree, list):
        return [_tree_map(fn, v) for v in tree]
    return tree


def _tree_reduce(fn, tree, init=0.0):
    """Reduce all arrays in a tree to a single value."""
    if isinstance(tree, mx.array):
        return fn(tree)
    elif isinstance(tree, dict):
        return sum(_tree_reduce(fn, v, init) for v in tree.values())
    elif isinstance(tree, list):
        return sum(_tree_reduce(fn, v, init) for v in tree)
    return init


def clip_grad_norm(grads, max_norm: float = 1.0):
    """Clip gradient tree by global norm.

    Returns the clipped grads and the original norm.
    """
    total_norm_sq = _tree_reduce(
        lambda g: mx.sum(g.astype(mx.float32) ** 2), grads
    )
    total_norm = mx.sqrt(total_norm_sq)
    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef = mx.minimum(clip_coef, mx.array(1.0))
    clipped = _tree_map(lambda g: g * clip_coef, grads)
    return clipped, total_norm


class BlockOptimizer:
    """Manages per-block optimizers for memory-efficient training.

    Each transformer block gets its own AdamW optimizer, so only the
    active block's optimizer state is needed during a training step.
    The shared parameters (embedding, LM head, timestep embedder, etc.)
    use a separate optimizer that is always active.

    Parameters
    ----------
    model : nn.Module
        The DBlockGPT model.
    num_blocks : int
        Number of transformer blocks.
    lr : float
        Learning rate.
    weight_decay : float
        AdamW weight decay.
    betas : tuple
        Adam beta parameters.
    warmup_steps : int
        Number of linear warmup steps.
    max_steps : int
        Total training steps (for cosine decay).  If 0, no decay.
    max_grad_norm : float
        Maximum gradient norm for clipping.  0 = no clipping.
    """

    def __init__(
        self,
        model: nn.Module,
        num_blocks: int,
        lr: float = 3e-4,
        weight_decay: float = 0.1,
        betas: tuple[float, float] = (0.9, 0.95),
        warmup_steps: int = 0,
        max_steps: int = 0,
        max_grad_norm: float = 1.0,
    ):
        self.num_blocks = num_blocks
        self.base_lr = lr
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.max_grad_norm = max_grad_norm
        self._step_count = 0

        # Create per-block optimizers
        self.block_optimizers = [
            optim.AdamW(
                learning_rate=lr,
                weight_decay=weight_decay,
                betas=betas,
            )
            for _ in range(num_blocks)
        ]

        # Shared parameter optimizer
        self.shared_optimizer = optim.AdamW(
            learning_rate=lr,
            weight_decay=weight_decay,
            betas=betas,
        )

    def _get_lr(self) -> float:
        """Compute current learning rate with warmup + cosine decay."""
        step = self._step_count
        if self.warmup_steps > 0 and step < self.warmup_steps:
            return self.base_lr * step / self.warmup_steps
        if self.max_steps > 0:
            progress = (step - self.warmup_steps) / max(
                1, self.max_steps - self.warmup_steps
            )
            progress = min(progress, 1.0)
            return self.base_lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))
        return self.base_lr

    def _set_lr(self, lr: float) -> None:
        """Update learning rate on all optimizers."""
        for opt in self.block_optimizers:
            opt.learning_rate = mx.array(lr)
        self.shared_optimizer.learning_rate = mx.array(lr)

    def step(
        self,
        model: nn.Module,
        grads: dict,
        block_idx: int,
    ) -> float:
        """Apply gradients to the active block and shared parameters.

        Parameters
        ----------
        model : nn.Module
            The model to update.
        grads : dict
            Gradient tree from ``nn.value_and_grad``.
        block_idx : int
            Index of the active block.

        Returns
        -------
        float
            Current learning rate.
        """
        self._step_count += 1
        lr = self._get_lr()
        self._set_lr(lr)

        # Clip gradients to prevent explosion from EDM weighting
        if self.max_grad_norm > 0:
            grads, grad_norm = clip_grad_norm(grads, self.max_grad_norm)
            mx.eval(grad_norm)

        # Apply gradients
        self.block_optimizers[block_idx].update(model, grads)

        # Evaluate to materialize updates and free computation graph
        mx.eval(model.parameters())
        for opt in [self.block_optimizers[block_idx]]:
            mx.eval(opt.state)

        return lr

    @property
    def step_count(self) -> int:
        return self._step_count
