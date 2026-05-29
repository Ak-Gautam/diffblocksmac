"""Training-time sampling and EDM coefficients.

Pure-Python helpers for sampling noise levels and computing the EDM
preconditioning / loss-weighting coefficients used by DiffusionBlocks.
"""

from __future__ import annotations

from math import log, sqrt
from random import Random
from typing import NamedTuple

from .blocks import BlockSpec
from .schedules import LogNormalNoise


class PrecondCoeffs(NamedTuple):
    """EDM preconditioning coefficients for a given sigma.

    Following Karras et al. (2022):
        c_skip  = σ_data² / (σ² + σ_data²)
        c_out   = σ · σ_data / √(σ² + σ_data²)
        c_in    = 1 / √(σ² + σ_data²)
        c_noise = 0.25 · ln(σ)
    """
    c_skip: float
    c_out: float
    c_in: float
    c_noise: float


def edm_weight(sigma: float, *, sigma_data: float = 0.5) -> float:
    """EDM loss weighting: ``(σ² + σ_data²) / (σ · σ_data)²``."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if sigma_data <= 0:
        raise ValueError("sigma_data must be positive")
    return (sigma ** 2 + sigma_data ** 2) / (sigma * sigma_data) ** 2


def preconditioning(
    sigma: float,
    *,
    sigma_data: float = 0.5,
) -> PrecondCoeffs:
    """Compute EDM preconditioning coefficients for *sigma*."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if sigma_data <= 0:
        raise ValueError("sigma_data must be positive")

    s2 = sigma ** 2
    d2 = sigma_data ** 2
    denom = sqrt(s2 + d2)
    return PrecondCoeffs(
        c_skip=d2 / (s2 + d2),
        c_out=sigma * sigma_data / denom,
        c_in=1.0 / denom,
        c_noise=0.25 * log(sigma),
    )


def sample_training_sigma(
    blocks: tuple[BlockSpec, ...] | list[BlockSpec],
    *,
    noise: LogNormalNoise | None = None,
    rng: Random | None = None,
) -> tuple[BlockSpec, float]:
    """Uniformly sample a block, then sample sigma from its expanded range.

    Returns ``(block, sigma)``.
    """
    if not blocks:
        raise ValueError("blocks must not be empty")

    noise = noise or LogNormalNoise()
    rng = rng or Random()

    block = blocks[rng.randrange(len(blocks))]
    cdf_lo = noise.cdf(block.train_sigma_min)
    cdf_hi = noise.cdf(block.train_sigma_max)
    p = rng.uniform(cdf_lo, cdf_hi)
    return block, noise.ppf(p)
