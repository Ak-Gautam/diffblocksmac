# DiffusionBlocks Paper Notes

Source paper: "DiffusionBlocks: Block-wise Neural Network Training via Diffusion
Interpretation" by Makoto Shing, Masanori Koyama, and Takuya Akiba. Local PDF:
`/Users/gautam/Downloads/2506.14202v3.pdf`. Web pages consulted:
<https://arxiv.org/abs/2506.14202> and <https://pub.sakana.ai/diffusionblocks/>.

## Core Idea

Standard end-to-end backpropagation stores activations for every layer. For a
deep residual/transformer model this makes training memory grow roughly
linearly with depth. DiffusionBlocks converts a residual network into a sequence
of independently trainable denoisers. Each denoiser/block owns one range of the
diffusion noise level, so training a step only needs one block's activations,
gradients, and optimizer state.

The paper's key observation is that residual updates look like Euler steps of a
continuous-time dynamical system. Diffusion probability-flow ODE sampling is
also implemented as repeated residual-style denoising updates. If a network's
blocks are reinterpreted as reverse diffusion steps from noise toward the target,
each block can be trained with a local score-matching/denoising objective.

## Conversion Recipe

1. Partition `L` layers into `B` contiguous blocks.
2. Choose a noise distribution and assign each block a sigma interval.
3. Add noise-level conditioning to each block, such as AdaLN/time embeddings.

At training time:

1. Sample a block.
2. Sample a target `z` and a noise level `sigma` from that block's interval.
3. Form `z_t = z + sigma * eps`.
4. Run only the selected block to predict/denoise the target.
5. Update only that block.

At inference time:

1. Start from noise.
2. Traverse sigma levels from high to low.
3. Select the block responsible for the current sigma.
4. Apply an Euler denoising update.

## Noise Partitioning

The paper recommends a log-normal training distribution, following EDM:

```text
log(sigma) ~ Normal(p_mean, p_std)
default: sigma_min=0.002, sigma_max=80, p_mean=-1.2, p_std=1.2
```

Blocks are assigned equal probability mass under the truncated log-normal
distribution, not equal width in sigma or log-sigma. Boundaries are:

```text
boundary_i = exp(p_mean + p_std * Phi^-1(p_i))
p_i = CDF(sigma_min) + (CDF(sigma_max) - CDF(sigma_min)) * i / B
```

This concentrates block boundaries around intermediate sigmas where the training
distribution has higher mass.

## Important Implementation Detail

The official code computes sigma intervals low-to-high, then reverses assignment
to model order. The first model block sees the highest sigma range during
inference, and the last model block sees the lowest sigma range. This repo's
`BlockSpec` stores both:

- `model_index`: forward/inference order through the model
- `noise_index`: low-to-high sigma interval index

## EDM Weighting And Preconditioning

The paper uses EDM weighting:

```text
w(sigma) = (sigma^2 + sigma_data^2) / (sigma * sigma_data)^2
default sigma_data = 0.5
```

The official implementation also uses EDM preconditioning:

```text
c_skip = sigma_data^2 / (sigma^2 + sigma_data^2)
c_out = sigma * sigma_data / sqrt(sigma^2 + sigma_data^2)
c_in = 1 / sqrt(sigma^2 + sigma_data^2)
c_noise = 0.25 * log(sigma)
```

## Block Overlap

To smooth transitions at block boundaries, each block's training sigma interval
is expanded in log-sigma space and clamped to the global range. The paper reports
an overlap around `0.1` as generally effective, and `0.2` for text generation.

## Architecture Adaptations

- ViT classification: add noise to class-label embeddings and train blocks to
  denoise the label representation conditioned on image patches. Use
  cross-entropy on the classifier output.
- DiT image generation: already denoising-native; partition sigma ranges across
  transformer blocks and use the relevant block per denoising step.
- Masked diffusion language models: partition the masking schedule by equal
  demasking work rather than partitioning raw time.
- Autoregressive language models: denoise token embeddings while preserving
  causal consistency, typically by letting noisy future tokens attend only to
  clean past tokens.
- Recurrent-depth models: train one denoising pass instead of BPTT through many
  recurrent iterations, while keeping iterative inference.

## Results To Reproduce Later

- ViT CIFAR-100: baseline 60.25 accuracy, DiffusionBlocks 59.30.
- DiT ImageNet-256: baseline test FID 12.09, DiffusionBlocks 10.63.
- Masked diffusion text8: baseline 1.56 BPC, DiffusionBlocks 1.45.
- AR Transformer OWT: baseline MAUVE 0.85, DiffusionBlocks 0.82.
- Recurrent-depth LM1B: baseline MAUVE 0.49, DiffusionBlocks 0.70.

## Library Direction

The first library layer should be backend-neutral math and scheduling. The next
layer should expose tensor backends. For this Mac-only project, the likely order
is:

1. PyTorch MPS backend for fastest path to correctness.
2. MLX backend for native Apple Silicon ergonomics and memory behavior.
3. Custom Metal/C++/Rust kernels only after profiling identifies a real bottleneck.
