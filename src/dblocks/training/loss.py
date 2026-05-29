"""Loss functions for DiffusionBlocks training.

Two loss variants:
- ``denoising_loss``: pure MSE denoising objective with EDM weighting
- ``ar_denoising_loss``: combines denoising + next-token cross-entropy
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn


def denoising_loss(
    z_pred: mx.array,
    z_noisy: mx.array,
    z_clean: mx.array,
    sigma: float | mx.array,
    *,
    sigma_data: float = 0.5,
) -> mx.array:
    """Compute EDM-weighted denoising loss.

    Applies preconditioning to the block's raw output and computes
    the weighted MSE between the preconditioned output and the clean target.

    Parameters
    ----------
    z_pred : mx.array
        Raw block output ``(B, T, D)`` — the block's prediction on ``c_in * z_noisy``.
    z_noisy : mx.array
        Noisy input ``(B, T, D)`` — before ``c_in`` scaling.
    z_clean : mx.array
        Clean target ``(B, T, D)``.
    sigma : float or mx.array
        Noise level (scalar or ``(B,)``).
    sigma_data : float
        EDM sigma_data parameter.

    Returns
    -------
    mx.array
        Scalar loss.
    """
    if isinstance(sigma, (int, float)):
        s2 = sigma ** 2
    else:
        s2 = sigma ** 2
        # Reshape for broadcasting: (B,) -> (B, 1, 1)
        while s2.ndim < z_pred.ndim:
            s2 = mx.expand_dims(s2, axis=-1)

    d2 = sigma_data ** 2

    # Preconditioning
    c_skip = d2 / (s2 + d2)
    c_out = (s2 ** 0.5) * sigma_data / (s2 + d2) ** 0.5  # σ·σ_d / √(σ²+σ_d²)

    # Preconditioned output
    z_out = c_skip * z_noisy + c_out * z_pred

    # EDM loss weight: (σ² + σ_d²) / (σ·σ_d)²
    weight = (s2 + d2) / (s2 * d2)

    # Weighted MSE
    diff = z_out - z_clean
    loss = weight * mx.mean(diff ** 2)
    return loss


def ar_denoising_loss(
    z_pred: mx.array,
    z_noisy: mx.array,
    z_clean: mx.array,
    logits: mx.array,
    targets: mx.array,
    sigma: float | mx.array,
    *,
    sigma_data: float = 0.5,
    lm_weight: float = 1.0,
) -> tuple[mx.array, dict[str, mx.array]]:
    """Combined denoising + autoregressive language model loss.

    This is the recommended loss for training GPT models with
    DiffusionBlocks.  It combines:
    1. EDM denoising loss (block learns to denoise representations)
    2. Cross-entropy LM loss (representations are useful for next-token prediction)

    Parameters
    ----------
    z_pred : mx.array
        Raw block output ``(B, T, D)``.
    z_noisy : mx.array
        Noisy input ``(B, T, D)``.
    z_clean : mx.array
        Clean target ``(B, T, D)``.
    logits : mx.array
        LM head output ``(B, T, vocab_size)``.
    targets : mx.array
        Next-token target IDs ``(B, T)``.
    sigma : float or mx.array
        Noise level.
    sigma_data : float
        EDM sigma_data.
    lm_weight : float
        Weight for the cross-entropy loss relative to denoising loss.

    Returns
    -------
    loss : mx.array
        Total scalar loss.
    metrics : dict
        Individual loss components for logging.
    """
    # Denoising component
    denoise_l = denoising_loss(z_pred, z_noisy, z_clean, sigma, sigma_data=sigma_data)

    # LM component: cross-entropy on next-token prediction
    # logits: (B, T, V), targets: (B, T)
    # Shift: predict token t+1 from position t
    shift_logits = logits[:, :-1, :]
    shift_targets = targets[:, 1:]

    # Flatten for cross-entropy
    B, T_1, V = shift_logits.shape
    flat_logits = shift_logits.reshape(-1, V)
    flat_targets = shift_targets.reshape(-1)

    lm_l = mx.mean(nn.losses.cross_entropy(flat_logits, flat_targets))

    total = denoise_l + lm_weight * lm_l

    metrics = {
        "loss": total,
        "denoise_loss": denoise_l,
        "lm_loss": lm_l,
    }
    return total, metrics
