"""Noise schedules used by DiffusionBlocks.

The defaults match the paper's EDM-style setup:
sigma_min=0.002, sigma_max=80, p_mean=-1.2, p_std=1.2.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, log
from statistics import NormalDist


_STANDARD_NORMAL = NormalDist()


@dataclass(frozen=True)
class LogNormalNoise:
    """Truncated log-normal noise distribution over sigma."""

    sigma_min: float = 0.002
    sigma_max: float = 80.0
    p_mean: float = -1.2
    p_std: float = 1.2

    def __post_init__(self) -> None:
        if self.sigma_min <= 0:
            raise ValueError("sigma_min must be positive")
        if self.sigma_max <= self.sigma_min:
            raise ValueError("sigma_max must be greater than sigma_min")
        if self.p_std <= 0:
            raise ValueError("p_std must be positive")

    def cdf(self, sigma: float) -> float:
        """CDF of log(sigma) under N(p_mean, p_std)."""

        if sigma <= 0:
            raise ValueError("sigma must be positive")
        return _STANDARD_NORMAL.cdf((log(sigma) - self.p_mean) / self.p_std)

    def ppf(self, probability: float) -> float:
        """Inverse CDF returning sigma."""

        if not 0.0 < probability < 1.0:
            raise ValueError("probability must be in the open interval (0, 1)")
        return exp(self.p_mean + self.p_std * _STANDARD_NORMAL.inv_cdf(probability))

    @property
    def cdf_min(self) -> float:
        return self.cdf(self.sigma_min)

    @property
    def cdf_max(self) -> float:
        return self.cdf(self.sigma_max)


def equi_probability_sigmas(
    num_blocks: int,
    *,
    noise: LogNormalNoise | None = None,
) -> tuple[float, ...]:
    """Return low-to-high sigma boundaries with equal truncated probability mass."""

    if num_blocks <= 0:
        raise ValueError("num_blocks must be positive")

    noise = noise or LogNormalNoise()
    cdf_min = noise.cdf_min
    cdf_max = noise.cdf_max
    boundaries = [
        noise.ppf(cdf_min + (cdf_max - cdf_min) * (i / num_blocks))
        for i in range(num_blocks + 1)
    ]
    boundaries[0] = noise.sigma_min
    boundaries[-1] = noise.sigma_max
    return tuple(boundaries)


def edm_sigmas(
    num_steps: int,
    *,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    rho: float = 7.0,
    descending: bool = True,
) -> tuple[float, ...]:
    """Karras/EDM power-law inference schedule."""

    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    if sigma_min <= 0:
        raise ValueError("sigma_min must be positive")
    if sigma_max <= sigma_min:
        raise ValueError("sigma_max must be greater than sigma_min")
    if rho <= 0:
        raise ValueError("rho must be positive")

    if num_steps == 1:
        sigmas = (sigma_max if descending else sigma_min,)
    else:
        min_inv_rho = sigma_min ** (1.0 / rho)
        max_inv_rho = sigma_max ** (1.0 / rho)
        values = []
        for i in range(num_steps):
            ramp = i / (num_steps - 1)
            sigma = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
            values.append(sigma)
        sigmas = tuple(values)

    return sigmas if descending else tuple(reversed(sigmas))


def dblock_inference_sigmas(
    num_steps: int,
    *,
    noise: LogNormalNoise | None = None,
    descending: bool = True,
) -> tuple[float, ...]:
    """Equi-probability sigma schedule used by DiffusionBlocks inference."""

    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    noise = noise or LogNormalNoise()

    if num_steps == 1:
        sigmas = (noise.sigma_max,)
    else:
        cdf_min = noise.cdf_min
        cdf_max = noise.cdf_max
        values = [
            noise.ppf(cdf_min + (cdf_max - cdf_min) * (i / (num_steps - 1)))
            for i in range(num_steps)
        ]
        values[0] = noise.sigma_min
        values[-1] = noise.sigma_max
        sigmas = tuple(values)

    return tuple(reversed(sigmas)) if descending else sigmas
