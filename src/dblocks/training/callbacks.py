"""Training callbacks for DiffusionBlocks."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Protocol

import mlx.core as mx


class Callback(Protocol):
    """Protocol for training callbacks."""

    def on_train_start(self, state: dict[str, Any]) -> None: ...
    def on_step_end(self, state: dict[str, Any]) -> None: ...
    def on_epoch_end(self, state: dict[str, Any]) -> None: ...
    def on_train_end(self, state: dict[str, Any]) -> None: ...


class ProgressCallback:
    """Print training progress to stdout.

    Parameters
    ----------
    log_every : int
        Print every N steps.
    """

    def __init__(self, log_every: int = 10):
        self.log_every = log_every
        self._start_time = 0.0
        self._step_times: list[float] = []

    def on_train_start(self, state: dict[str, Any]) -> None:
        self._start_time = time.time()
        total_params = state.get("total_params", 0)
        num_blocks = state.get("num_blocks", 0)
        mem = state.get("memory_estimate", {})
        print(f"\n{'='*60}")
        print(f"DiffusionBlocks Training")
        print(f"{'='*60}")
        print(f"Parameters: {total_params:,}")
        print(f"Blocks: {num_blocks}")
        if mem:
            print(f"Params memory:     {mem.get('params', 0):.1f} MB")
            print(f"E2E peak estimate: {mem.get('e2e_peak_estimate', 0):.1f} MB")
            print(f"DBlock peak est:   {mem.get('dblock_peak_estimate', 0):.1f} MB")
            savings = mem.get('e2e_peak_estimate', 1) / max(mem.get('dblock_peak_estimate', 1), 0.1)
            print(f"Memory reduction:  ~{savings:.1f}x")
        print(f"{'='*60}\n")

    def on_step_end(self, state: dict[str, Any]) -> None:
        step = state["step"]
        self._step_times.append(time.time())
        if step % self.log_every == 0 or step == 1:
            loss = state.get("loss", 0.0)
            lr = state.get("lr", 0.0)
            block_idx = state.get("block_idx", -1)
            sigma = state.get("sigma", 0.0)
            elapsed = time.time() - self._start_time
            metrics = state.get("metrics", {})

            parts = [
                f"step {step:>6d}",
                f"block {block_idx}",
                f"σ={sigma:.3f}",
                f"loss={loss:.4f}",
            ]
            if "denoise_loss" in metrics:
                parts.append(f"denoise={float(metrics['denoise_loss']):.4f}")
            if "lm_loss" in metrics:
                parts.append(f"lm={float(metrics['lm_loss']):.4f}")
            parts.append(f"lr={lr:.2e}")
            parts.append(f"t={elapsed:.1f}s")

            # Steps per second
            if len(self._step_times) >= 2:
                recent = self._step_times[-min(10, len(self._step_times)):]
                sps = len(recent) / (recent[-1] - recent[0] + 1e-8)
                parts.append(f"{sps:.1f} steps/s")

            print(" | ".join(parts))

    def on_epoch_end(self, state: dict[str, Any]) -> None:
        epoch = state.get("epoch", 0)
        avg_loss = state.get("epoch_avg_loss", 0.0)
        elapsed = time.time() - self._start_time
        print(f"\n--- Epoch {epoch} complete | avg_loss={avg_loss:.4f} | {elapsed:.1f}s ---\n")

    def on_train_end(self, state: dict[str, Any]) -> None:
        elapsed = time.time() - self._start_time
        print(f"\nTraining complete in {elapsed:.1f}s")


class MemoryCallback:
    """Log memory usage during training.

    Parameters
    ----------
    log_every : int
        Log every N steps.
    """

    def __init__(self, log_every: int = 50):
        self.log_every = log_every

    def on_train_start(self, state: dict[str, Any]) -> None:
        self._report("start")

    def on_step_end(self, state: dict[str, Any]) -> None:
        if state["step"] % self.log_every == 0:
            self._report(f"step {state['step']}")

    def on_epoch_end(self, state: dict[str, Any]) -> None:
        pass

    def on_train_end(self, state: dict[str, Any]) -> None:
        self._report("end")

    @staticmethod
    def _report(label: str) -> None:
        def _get(name: str) -> float:
            try:
                return getattr(mx, name)() / (1024 ** 2)
            except (AttributeError, Exception):
                try:
                    return getattr(mx.metal, name)() / (1024 ** 2)
                except Exception:
                    return 0.0
        try:
            active = _get("get_active_memory")
            peak = _get("get_peak_memory")
            cache = _get("get_cache_memory")
            print(
                f"[mem@{label}] active={active:.1f}MB "
                f"peak={peak:.1f}MB cache={cache:.1f}MB"
            )
        except Exception:
            pass


class CheckpointCallback:
    """Save model checkpoints during training.

    Parameters
    ----------
    save_dir : str or Path
        Directory to save checkpoints.
    save_every : int
        Save every N epochs.
    keep_last : int
        Keep only the last N checkpoints (0 = keep all).
    """

    def __init__(
        self,
        save_dir: str | Path = "checkpoints",
        save_every: int = 1,
        keep_last: int = 3,
    ):
        self.save_dir = Path(save_dir)
        self.save_every = save_every
        self.keep_last = keep_last
        self._saved: list[Path] = []

    def on_train_start(self, state: dict[str, Any]) -> None:
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def on_step_end(self, state: dict[str, Any]) -> None:
        pass

    def on_epoch_end(self, state: dict[str, Any]) -> None:
        epoch = state.get("epoch", 0)
        if epoch % self.save_every != 0:
            return

        model = state.get("model")
        if model is None:
            return

        path = self.save_dir / f"epoch_{epoch:04d}"
        path.mkdir(exist_ok=True)

        # Save model weights
        weights_path = path / "weights.npz"
        mx.savez(str(weights_path), **_flatten_params(model.parameters()))

        # Save config if available
        config = getattr(model, "config", None)
        if config is not None:
            import dataclasses
            config_dict = dataclasses.asdict(config)
            # Convert non-serializable fields
            config_dict.pop("noise", None)
            config_dict["dtype"] = str(config.dtype)
            with open(path / "config.json", "w") as f:
                json.dump(config_dict, f, indent=2)

        # Save training state
        train_state = {
            "epoch": epoch,
            "step": state.get("step", 0),
            "loss": float(state.get("epoch_avg_loss", 0)),
        }
        with open(path / "train_state.json", "w") as f:
            json.dump(train_state, f, indent=2)

        self._saved.append(path)
        print(f"Checkpoint saved: {path}")

        # Cleanup old checkpoints
        if self.keep_last > 0 and len(self._saved) > self.keep_last:
            old = self._saved.pop(0)
            import shutil
            shutil.rmtree(old, ignore_errors=True)

    def on_train_end(self, state: dict[str, Any]) -> None:
        pass


def _flatten_params(params: dict, prefix: str = "") -> dict[str, mx.array]:
    """Flatten nested parameter dict for saving."""
    flat = {}
    for k, v in params.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, mx.array):
            flat[key] = v
        elif isinstance(v, dict):
            flat.update(_flatten_params(v, key))
        elif isinstance(v, list):
            for i, item in enumerate(v):
                if isinstance(item, mx.array):
                    flat[f"{key}.{i}"] = item
                elif isinstance(item, dict):
                    flat.update(_flatten_params(item, f"{key}.{i}"))
    return flat
