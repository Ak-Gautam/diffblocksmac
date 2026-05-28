"""Block partitioning utilities."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, log
from typing import Iterable

from .schedules import LogNormalNoise, equi_probability_sigmas


@dataclass(frozen=True)
class BlockSpec:
    """A model block and its assigned noise interval.

    `model_index` follows the forward/inference order of the network:
    index 0 is the earliest block and receives the highest sigma range.

    `noise_index` follows low-to-high sigma intervals from the truncated
    log-normal distribution. The mapping is reversed:
    `model_index = num_blocks - 1 - noise_index`.
    """

    model_index: int
    noise_index: int
    layers: tuple[int, ...]
    sigma_min: float
    sigma_max: float
    train_sigma_min: float
    train_sigma_max: float

    def contains(self, sigma: float, *, training: bool = False) -> bool:
        lo = self.train_sigma_min if training else self.sigma_min
        hi = self.train_sigma_max if training else self.sigma_max
        return lo <= sigma <= hi


def partition_layers(
    num_layers: int,
    num_blocks: int,
    *,
    layer_counts: Iterable[int] | None = None,
) -> tuple[tuple[int, ...], ...]:
    """Split contiguous layers into model-order blocks."""

    if num_layers <= 0:
        raise ValueError("num_layers must be positive")
    if num_blocks <= 0:
        raise ValueError("num_blocks must be positive")
    if num_blocks > num_layers:
        raise ValueError("num_blocks cannot exceed num_layers")

    if layer_counts is None:
        base = num_layers // num_blocks
        rem = num_layers % num_blocks
        counts = [base + (1 if i < rem else 0) for i in range(num_blocks)]
    else:
        counts = list(layer_counts)
        if len(counts) != num_blocks:
            raise ValueError("layer_counts length must match num_blocks")
        if any(count <= 0 for count in counts):
            raise ValueError("layer_counts must all be positive")
        if sum(counts) != num_layers:
            raise ValueError("layer_counts must sum to num_layers")

    layers = []
    start = 0
    for count in counts:
        stop = start + count
        layers.append(tuple(range(start, stop)))
        start = stop
    return tuple(layers)


def _expand_sigma_range(
    sigma_min: float,
    sigma_max: float,
    *,
    global_sigma_min: float,
    global_sigma_max: float,
    overlap: float,
) -> tuple[float, float]:
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap == 0:
        return sigma_min, sigma_max

    lo = log(sigma_min)
    hi = log(sigma_max)
    width = hi - lo
    expanded_min = exp(lo - overlap * width)
    expanded_max = exp(hi + overlap * width)
    return max(expanded_min, global_sigma_min), min(expanded_max, global_sigma_max)


def make_block_specs(
    num_layers: int,
    num_blocks: int,
    *,
    noise: LogNormalNoise | None = None,
    overlap: float = 0.1,
    layer_counts: Iterable[int] | None = None,
) -> tuple[BlockSpec, ...]:
    """Build model-order block specs.

    The paper's implementation samples sigma intervals low-to-high but assigns
    them to model layers high-to-low. This function returns specs in model
    order, so `specs[0]` is the first network block and the highest-noise block.
    """

    noise = noise or LogNormalNoise()
    layers = partition_layers(num_layers, num_blocks, layer_counts=layer_counts)
    boundaries = equi_probability_sigmas(num_blocks, noise=noise)
    specs = []

    for model_index in range(num_blocks):
        noise_index = num_blocks - 1 - model_index
        sigma_min = boundaries[noise_index]
        sigma_max = boundaries[noise_index + 1]
        train_sigma_min, train_sigma_max = _expand_sigma_range(
            sigma_min,
            sigma_max,
            global_sigma_min=noise.sigma_min,
            global_sigma_max=noise.sigma_max,
            overlap=overlap,
        )
        specs.append(
            BlockSpec(
                model_index=model_index,
                noise_index=noise_index,
                layers=layers[model_index],
                sigma_min=sigma_min,
                sigma_max=sigma_max,
                train_sigma_min=train_sigma_min,
                train_sigma_max=train_sigma_max,
            )
        )

    return tuple(specs)
