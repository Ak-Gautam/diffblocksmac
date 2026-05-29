# dblocks

`dblocks` is a Mac-first Python library for experimenting with DiffusionBlocks:
block-wise neural network training by interpreting residual blocks as denoising
steps.

The repository is starting from the paper-critical primitives first:

- equi-probability log-normal noise partitioning
- block-to-layer assignment with high-noise blocks first
- overlap-expanded block ranges
- EDM weighting and preconditioning coefficients
- backend-neutral APIs that can later target PyTorch MPS, MLX, or custom Metal
  kernels

The package name is intentionally short:

```python
import dblocks as db
```

## Current Status

This is an initial scaffold, not yet a full neural-network framework. The next
implementation step is a backend interface and a minimal trainable transformer
example on Apple Silicon.

## Design Constraints

- macOS and Apple Silicon are the target platform.
- Python is the default user-facing language.
- Low-level kernels are allowed where profiling justifies them.
- Correctness comes first, followed by memory/time efficiency, then ergonomics.
