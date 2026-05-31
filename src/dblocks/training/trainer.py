"""DiffusionBlocks trainer — the main training loop.

Implements the paper's block-wise training algorithm optimized for
Apple Silicon unified memory:

1. Sample a block index uniformly
2. Sample sigma from the block's truncated noise range
3. Freeze all blocks except the active one
4. Add noise to clean embeddings
5. Forward only the active block
6. Compute denoising + LM loss
7. Backward through the active block only
8. Update the active block's parameters
"""

from __future__ import annotations

from random import Random
from typing import Any, Iterator

import mlx.core as mx
import mlx.nn as nn

from ..core.sampling import sample_training_sigmas
from ..models.gpt import DBlockGPT
from ..nn.attention import _create_causal_mask
from .callbacks import Callback, ProgressCallback
from .optimizer import BlockOptimizer


class DBlockTrainer:
    """Memory-efficient block-wise trainer for DiffusionBlocks.

    Parameters
    ----------
    model : DBlockGPT
        The model to train.
    lr : float
        Peak learning rate.
    weight_decay : float
        AdamW weight decay.
    warmup_steps : int
        Linear warmup steps.
    max_steps : int
        Total training steps (for cosine decay). 0 = no decay.
    lm_weight : float
        Weight for the cross-entropy loss component.
    metal_cache_limit : int
        MLX Metal cache limit in bytes.  Lower = more aggressive memory
        freeing.  Default 512 MB.
    callbacks : list
        Training callbacks.
    seed : int
        Random seed.
    """

    def __init__(
        self,
        model: DBlockGPT,
        lr: float = 3e-4,
        weight_decay: float = 0.1,
        warmup_steps: int = 100,
        max_steps: int = 0,
        lm_weight: float = 1.0,
        metal_cache_limit: int = 512 * 1024 * 1024,
        callbacks: list[Callback] | None = None,
        seed: int = 42,
    ):
        self.model = model
        self.lm_weight = lm_weight
        self.callbacks = callbacks or [ProgressCallback()]

        # Set Metal memory constraints
        try:
            mx.set_cache_limit(metal_cache_limit)
        except Exception:
            try:
                mx.metal.set_cache_limit(metal_cache_limit)
            except Exception:
                pass

        # Random state
        mx.random.seed(seed)
        self._rng = Random(seed)

        # Create block optimizer
        total_steps = max_steps
        self.optimizer = BlockOptimizer(
            model,
            num_blocks=model.num_blocks,
            lr=lr,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            max_steps=total_steps,
        )

        # Noise distribution
        self._noise = model.config.noise
        self._sigma_data = model.config.sigma_data

        # Build the loss + grad function
        # This is created once and reused — MLX compiles it efficiently.
        self._loss_and_grad_fn = nn.value_and_grad(
            model, self._compute_loss
        )

    def _compute_loss(
        self,
        model: DBlockGPT,
        tokens: mx.array,
        block_idx: int,
        sigmas: mx.array,
    ) -> mx.array:
        """Compute combined denoising + LM loss for a single block.

        This function is traced by MLX's autograd — it must be a pure
        function of the model parameters and inputs.

        All intermediate tensor math is done in float32 to prevent
        overflow from EDM weighting (σ can be up to 80, and the weight
        is O(σ²) which exceeds float16 max of 65504).  Model weights
        stay in float16; MLX handles mixed-precision matmul natively.
        """
        B, T = tokens.shape
        sd = self._sigma_data

        # 1. Get clean embeddings — upcast to float32 for all math
        z_clean = model.embed(tokens).astype(mx.float32)

        sigmas = sigmas.astype(mx.float32)
        s = sigmas[:, None, None]

        # 2. Add noise (in float32; sigma can be up to 80)
        noise = mx.random.normal(z_clean.shape, dtype=mx.float32)
        z_noisy = z_clean + s * noise

        # 3. EDM preconditioning, per example
        s2 = sigmas ** 2
        d2 = sd ** 2
        denom = (s2 + d2) ** 0.5
        c_in = (1.0 / denom)[:, None, None]
        c_skip = (d2 / (s2 + d2))[:, None, None]
        c_out = (sigmas * sd / denom)[:, None, None]
        c_noise = 0.25 * mx.log(sigmas)

        # 4. Conditioning (float32 input)
        cond = model.t_embed(c_noise)
        if cond.dtype != mx.float32:
            cond = cond.astype(mx.float32)

        # 5. Causal mask
        mask = _create_causal_mask(T, dtype=mx.float32)

        # 6. Forward through active block only (float32 activations)
        z_pred = model.forward_block(c_in * z_noisy, block_idx, cond, mask)
        if z_pred.dtype != mx.float32:
            z_pred = z_pred.astype(mx.float32)

        # 7. Preconditioned output
        z_out = c_skip * z_noisy + c_out * z_pred

        # 8. Denoising loss (EDM weighted)
        weight = (s2 + d2) / (s2 * d2)
        per_example_mse = mx.mean((z_out - z_clean).reshape(B, -1) ** 2, axis=1)
        denoise_loss = mx.mean(weight * per_example_mse)

        # 9. LM loss from preconditioned output
        logits = model.lm_head(model.ln_f(z_out))
        if logits.dtype != mx.float32:
            logits = logits.astype(mx.float32)
        shift_logits = logits[:, :-1, :].reshape(-1, logits.shape[-1])
        shift_targets = tokens[:, 1:].reshape(-1)
        lm_loss = mx.mean(nn.losses.cross_entropy(shift_logits, shift_targets))

        return denoise_loss + self.lm_weight * lm_loss

    def train_step(self, tokens: mx.array) -> dict[str, Any]:
        """Execute one DiffusionBlocks training step.

        Parameters
        ----------
        tokens : mx.array
            Token IDs ``(B, T)``.

        Returns
        -------
        dict
            Step metrics including loss, block_idx, sigma, lr.
        """
        # 1. Sample block and sigma
        block, sigma_values = sample_training_sigmas(
            self.model.block_specs,
            tokens.shape[0],
            noise=self._noise,
            rng=self._rng,
        )
        block_idx = block.model_index
        sigmas = mx.array(sigma_values, dtype=mx.float32)

        # 2. Freeze all except active block + shared
        self.model.freeze_for_block(block_idx)

        # 3. Compute loss and gradients
        loss, grads = self._loss_and_grad_fn(
            self.model, tokens, block_idx, sigmas,
        )

        # 4. Optimizer step
        lr = self.optimizer.step(self.model, grads, block_idx)

        # 5. Force evaluation to free memory
        mx.eval(loss)

        return {
            "loss": float(loss),
            "block_idx": block_idx,
            "sigma": float(mx.mean(sigmas)),
            "sigma_min": min(sigma_values),
            "sigma_max": max(sigma_values),
            "lr": lr,
            "metrics": {},
        }

    def fit(
        self,
        train_data: Iterator[mx.array],
        *,
        epochs: int = 1,
        steps_per_epoch: int | None = None,
    ) -> list[dict[str, Any]]:
        """Run the full training loop.

        Parameters
        ----------
        train_data : Iterator
            Yields batches of token IDs ``(B, T)``.
            Must be re-iterable (called once per epoch) or infinite.
        epochs : int
            Number of training epochs.
        steps_per_epoch : int, optional
            Number of steps per epoch.  If None, iterate until data is
            exhausted.

        Returns
        -------
        list
            Per-step metrics.
        """
        history: list[dict[str, Any]] = []

        # Training start callback
        state = {
            "total_params": self.model.num_parameters(),
            "num_blocks": self.model.num_blocks,
            "memory_estimate": self.model.memory_estimate_mb(),
        }
        for cb in self.callbacks:
            cb.on_train_start(state)

        try:
            mx.reset_peak_memory()
        except Exception:
            try:
                mx.metal.reset_peak_memory()
            except Exception:
                pass

        global_step = 0

        for epoch in range(1, epochs + 1):
            epoch_losses: list[float] = []
            step_in_epoch = 0

            for batch in train_data:
                global_step += 1
                step_in_epoch += 1

                # Ensure batch is on the right dtype
                if batch.dtype != mx.int32 and batch.dtype != mx.uint32:
                    batch = batch.astype(mx.int32)

                result = self.train_step(batch)
                result["step"] = global_step
                result["epoch"] = epoch
                epoch_losses.append(result["loss"])
                history.append(result)

                # Step callback
                for cb in self.callbacks:
                    cb.on_step_end(result)

                if steps_per_epoch is not None and step_in_epoch >= steps_per_epoch:
                    break

            # Epoch callback
            epoch_state = {
                "epoch": epoch,
                "step": global_step,
                "epoch_avg_loss": sum(epoch_losses) / max(len(epoch_losses), 1),
                "model": self.model,
            }
            for cb in self.callbacks:
                cb.on_epoch_end(epoch_state)

        # Training end callback
        for cb in self.callbacks:
            cb.on_train_end({"step": global_step, "model": self.model})

        return history
