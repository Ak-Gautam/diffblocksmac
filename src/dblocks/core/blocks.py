"""Block partitioning utilities.

A DiffusionBlocks model has *B* contiguous blocks, each assigned a sigma
interval from the log-normal noise distribution.  The first model block
(``model_index=0``) processes the **highest** sigma range; the last block
(``model_index=B-1``) processes the **lowest**.  This matches the paper's
convention: early layers handle coarse/noisy representations and later
layers handle fine/clean ones.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, log
from typing import Iterable

from .schedules import LogNormalNoise, equi_probability_sigmas


@dataclass(frozen=True)
class BlockSpec:
    """Specification for one trainable block.

    Attributes
    ----------
    model_index : int
        Position in the forward pass (0 = earliest block).
    noise_index : int
        Low-to-high sigma interval index from the noise distribution.
        ``model_index = num_blocks - 1 - noise_index``.
    layers : tuple[int, ...]
        Layer indices assigned to this block.
    sigma_min, sigma_max : float
        Exact boundaries of the block's sigma interval.
    train_sigma_min, train_sigma_max : float
        Expanded boundaries used during training (overlap).
    """

    model_index: int
    noise_index: int
    layers: tuple[int, ...]
    sigma_min: float
    sigma_max: float
    train_sigma_min: float
    train_sigma_max: float

    @property
    def sigma_mid(self) -> float:
        """Geometric midpoint of the sigma interval (useful for inference)."""
        return exp(0.5 * (log(self.sigma_min) + log(self.sigma_max)))

    def contains(self, sigma: float, *, training: bool = False) -> bool:
        lo = self.train_sigma_min if training else self.sigma_min
        hi = self.train_sigma_max if training else self.sigma_max
        return lo <= sigma <= hi


# --------------------------------------------------------------------------
# Layer partitioning
# --------------------------------------------------------------------------

def partition_layers(
    num_layers: int,
    num_blocks: int,
    *,
    layer_counts: Iterable[int] | None = None,
) -> tuple[tuple[int, ...], ...]:
    """Split ``num_layers`` contiguous layer indices into *model-order* blocks.

    By default layers are split as evenly as possible.  Provide
    ``layer_counts`` to override.
    """
    if num_layers <= 0:
        raise ValueError("num_layers must be positive")
    if num_blocks <= 0:
        raise ValueError("num_blocks must be positive")
    if num_blocks > num_layers:
        raise ValueError("num_blocks cannot exceed num_layers")

    if layer_counts is None:
        base, rem = divmod(num_layers, num_blocks)
        counts = [base + (1 if i < rem else 0) for i in range(num_blocks)]
    else:
        counts = list(layer_counts)
        if len(counts) != num_blocks:
            raise ValueError("layer_counts length must match num_blocks")
        if any(c <= 0 for c in counts):
            raise ValueError("all layer_counts must be positive")
        if sum(counts) != num_layers:
            raise ValueError("layer_counts must sum to num_layers")

    groups: list[tuple[int, ...]] = []
    start = 0
    for c in counts:
        groups.append(tuple(range(start, start + c)))
        start += c
    return tuple(groups)


# --------------------------------------------------------------------------
# Sigma range expansion (overlap)
# --------------------------------------------------------------------------

def _expand_sigma_range(
    s_min: float,
    s_max: float,
    *,
    global_min: float,
    global_max: float,
    overlap: float,
) -> tuple[float, float]:
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap == 0:
        return s_min, s_max
    lo, hi = log(s_min), log(s_max)
    width = hi - lo
    return (
        max(exp(lo - overlap * width), global_min),
        min(exp(hi + overlap * width), global_max),
    )


# --------------------------------------------------------------------------
# Block spec factory
# --------------------------------------------------------------------------

def make_block_specs(
    num_layers: int,
    num_blocks: int,
    *,
    noise: LogNormalNoise | None = None,
    overlap: float = 0.1,
    layer_counts: Iterable[int] | None = None,
) -> tuple[BlockSpec, ...]:
    """Build ``num_blocks`` :class:`BlockSpec` objects in **model order**.

    ``specs[0]`` is the first network block (highest sigma range).
    ``specs[-1]`` is the last network block (lowest sigma range).
    """
    noise = noise or LogNormalNoise()
    layers = partition_layers(num_layers, num_blocks, layer_counts=layer_counts)
    boundaries = equi_probability_sigmas(num_blocks, noise=noise)
    specs: list[BlockSpec] = []

    for model_idx in range(num_blocks):
        noise_idx = num_blocks - 1 - model_idx
        s_min = boundaries[noise_idx]
        s_max = boundaries[noise_idx + 1]
        t_min, t_max = _expand_sigma_range(
            s_min, s_max,
            global_min=noise.sigma_min,
            global_max=noise.sigma_max,
            overlap=overlap,
        )
        specs.append(BlockSpec(
            model_index=model_idx,
            noise_index=noise_idx,
            layers=layers[model_idx],
            sigma_min=s_min,
            sigma_max=s_max,
            train_sigma_min=t_min,
            train_sigma_max=t_max,
        ))

    return tuple(specs)
