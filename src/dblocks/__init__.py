"""dblocks — Memory-efficient DiffusionBlocks training for Apple Silicon.

DiffusionBlocks partitions a transformer into independently trainable
blocks, each conditioned on a noise level and trained with a denoising
objective.  This reduces peak memory proportional to the number of blocks,
enabling training of larger models on constrained hardware.

Quick start::

    import dblocks as db

    # Create a small GPT model with 3 DiffusionBlocks
    config = db.GPTConfig.tiny(vocab_size=256, num_blocks=2)
    model = db.DBlockGPT(config)

    # Create dataset and trainer
    ds, batcher = db.data.create_dataset("your text here...", seq_len=64)
    trainer = db.DBlockTrainer(model, lr=3e-4)

    # Train
    trainer.fit(batcher, epochs=1, steps_per_epoch=100)
"""

__version__ = "0.1.0"

# -- Core math (pure Python, no tensor dependency) -------------------------
from .core import (
    BlockSpec,
    LogNormalNoise,
    PrecondCoeffs,
    dblock_inference_sigmas,
    edm_sigmas,
    edm_weight,
    equi_probability_sigmas,
    make_block_specs,
    partition_layers,
    preconditioning,
    sample_training_sigma,
)

# -- Models ----------------------------------------------------------------
from .models import DBlockGPT, GPTConfig

# -- Training engine -------------------------------------------------------
from .training import (
    BlockOptimizer,
    CheckpointCallback,
    DBlockTrainer,
    MemoryCallback,
    ProgressCallback,
)

# -- Memory ----------------------------------------------------------------
from .memory import MemoryProfiler, memory_summary

# -- Data ------------------------------------------------------------------
from . import data

__all__ = [
    # Core
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
    # Models
    "DBlockGPT",
    "GPTConfig",
    # Training
    "BlockOptimizer",
    "CheckpointCallback",
    "DBlockTrainer",
    "MemoryCallback",
    "ProgressCallback",
    # Memory
    "MemoryProfiler",
    "memory_summary",
    # Data
    "data",
]
