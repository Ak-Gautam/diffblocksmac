"""Training-time sampling and EDM coefficients."""

from __future__ import annotations

from dataclasses import dataclass
from math import log
from random import Random

from .blocks import BlockSpec
from .schedules import LogNormalNoise


@dataclass(frozen=True)
class TrainingSample:
    block: BlockSpec
    sigma: float


@dataclass(frozen=True)
class PreconditioningCoefficients:
    c_skip: float
    c_out: float
    c_in: float
    c_noise: float


def edm_weight(sigma: float, *, sigma_data: float = 0.5) -> float:
    """EDM loss weighting used by the paper."""

    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if sigma_data <= 0:
        raise ValueError("sigma_data must be positive")
    return (sigma**2 + sigma_data**2) / ((sigma * sigma_data) ** 2)


def preconditioning_coefficients(
    sigma: float,
    *,
    sigma_data: float = 0.5,
) -> PreconditioningCoefficients:
    """EDM preconditioning coefficients used around each denoiser."""

    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if sigma_data <= 0:
        raise ValueError("sigma_data must be positive")

    denom = (sigma**2 + sigma_data**2) ** 0.5
    return PreconditioningCoefficients(
        c_skip=sigma_data**2 / (sigma**2 + sigma_data**2),
        c_out=sigma * sigma_data / denom,
        c_in=1.0 / denom,
        c_noise=0.25 * log(sigma),
    )


def sample_training_sigma(
    blocks: tuple[BlockSpec, ...],
    *,
    noise: LogNormalNoise | None = None,
    rng: Random | None = None,
) -> TrainingSample:
    """Uniformly sample one block, then sample sigma from its truncated range."""

    if not blocks:
        raise ValueError("blocks must not be empty")

    noise = noise or LogNormalNoise()
    rng = rng or Random()
    block = blocks[rng.randrange(len(blocks))]

    cdf_min = noise.cdf(block.train_sigma_min)
    cdf_max = noise.cdf(block.train_sigma_max)
    probability = rng.uniform(cdf_min, cdf_max)
    return TrainingSample(block=block, sigma=noise.ppf(probability))
