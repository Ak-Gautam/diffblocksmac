"""Core DiffusionBlocks math primitives.

Pure-Python module with no tensor framework dependency.  Used for noise
schedule computation, block partitioning, and EDM coefficient calculation.
"""

from .blocks import BlockSpec, make_block_specs, partition_layers
from .sampling import (
    PrecondCoeffs,
    edm_weight,
    preconditioning,
    sample_training_sigma,
)
from .schedules import (
    LogNormalNoise,
    dblock_inference_sigmas,
    edm_sigmas,
    equi_probability_sigmas,
)

__all__ = [
    "BlockSpec",
    "LogNormalNoise",
    "PrecondCoeffs",
    "dblock_inference_sigmas",
    "edm_sigmas",
    "edm_weight",
    "equi_probability_sigmas",
    "make_block_specs",
    "partition_layers",
    "preconditioning",
    "sample_training_sigma",
]
