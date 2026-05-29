"""Noise schedules for DiffusionBlocks.

Defaults match the paper's EDM-style setup:
    sigma_min=0.002, sigma_max=80, p_mean=-1.2, p_std=1.2

All functions in this module are pure Python (no tensor dependency).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, log
from statistics import NormalDist

_STD_NORMAL = NormalDist()


@dataclass(frozen=True)
class LogNormalNoise:
    """Truncated log-normal noise distribution over sigma.

    The training distribution is ``log(sigma) ~ N(p_mean, p_std)`` truncated
    to ``[sigma_min, sigma_max]``.  Block boundaries are placed at quantiles
    of equal probability mass under this distribution.
    """

    sigma_min: float = 0.002
    sigma_max: float = 80.0
    p_mean: float = -1.2
    p_std: float = 1.2

    def __post_init__(self) -> None:
        if self.sigma_min <= 0:
            raise ValueError("sigma_min must be positive")
        if self.sigma_max <= self.sigma_min:
            raise ValueError("sigma_max must exceed sigma_min")
        if self.p_std <= 0:
            raise ValueError("p_std must be positive")

    # -- CDF / inverse CDF ------------------------------------------------

    def cdf(self, sigma: float) -> float:
        """CDF of ``log(sigma)`` under ``N(p_mean, p_std)``."""
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        return _STD_NORMAL.cdf((log(sigma) - self.p_mean) / self.p_std)

    def ppf(self, p: float) -> float:
        """Inverse CDF – return sigma for a given cumulative probability."""
        if not 0.0 < p < 1.0:
            raise ValueError("p must be in the open interval (0, 1)")
        return exp(self.p_mean + self.p_std * _STD_NORMAL.inv_cdf(p))

    # -- Convenience -------------------------------------------------------

    @property
    def cdf_min(self) -> float:
        return self.cdf(self.sigma_min)

    @property
    def cdf_max(self) -> float:
        return self.cdf(self.sigma_max)


# --------------------------------------------------------------------------
# Schedule generators
# --------------------------------------------------------------------------

def equi_probability_sigmas(
    num_blocks: int,
    *,
    noise: LogNormalNoise | None = None,
) -> tuple[float, ...]:
    """Return ``num_blocks + 1`` sigma boundaries with equal probability mass.

    Boundaries are sorted low-to-high.  ``boundaries[0] == sigma_min`` and
    ``boundaries[-1] == sigma_max``.
    """
    if num_blocks <= 0:
        raise ValueError("num_blocks must be positive")

    noise = noise or LogNormalNoise()
    lo, hi = noise.cdf_min, noise.cdf_max
    boundaries = [
        noise.ppf(lo + (hi - lo) * (i / num_blocks))
        for i in range(num_blocks + 1)
    ]
    # Pin endpoints exactly to avoid float drift.
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
    """Karras / EDM power-law inference schedule."""
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    if sigma_min <= 0:
        raise ValueError("sigma_min must be positive")
    if sigma_max <= sigma_min:
        raise ValueError("sigma_max must exceed sigma_min")
    if rho <= 0:
        raise ValueError("rho must be positive")

    if num_steps == 1:
        vals = (sigma_max if descending else sigma_min,)
    else:
        lo_r = sigma_min ** (1.0 / rho)
        hi_r = sigma_max ** (1.0 / rho)
        vals = tuple(
            (hi_r + i / (num_steps - 1) * (lo_r - hi_r)) ** rho
            for i in range(num_steps)
        )
    return vals if descending else tuple(reversed(vals))


def dblock_inference_sigmas(
    num_steps: int,
    *,
    noise: LogNormalNoise | None = None,
    descending: bool = True,
) -> tuple[float, ...]:
    """Equi-probability sigma schedule for DiffusionBlocks inference."""
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    noise = noise or LogNormalNoise()

    if num_steps == 1:
        vals = (noise.sigma_max,)
    else:
        lo, hi = noise.cdf_min, noise.cdf_max
        vals_list = [
            noise.ppf(lo + (hi - lo) * (i / (num_steps - 1)))
            for i in range(num_steps)
        ]
        vals_list[0] = noise.sigma_min
        vals_list[-1] = noise.sigma_max
        vals = tuple(vals_list)

    return tuple(reversed(vals)) if descending else vals
