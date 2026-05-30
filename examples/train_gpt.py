#!/usr/bin/env python3
"""Train a tiny GPT with DiffusionBlocks on Apple Silicon.

This example demonstrates the full training pipeline:
1. Create a character-level dataset from synthetic text
2. Build a small DBlockGPT model
3. Train with the DiffusionBlocks block-wise algorithm
4. Show memory savings vs. hypothetical end-to-end training
5. Generate sample text

Usage:
    uv run python examples/train_gpt.py
    uv run python examples/train_gpt.py --epochs 5 --steps 200
"""

from __future__ import annotations

import argparse
import math
import os
import time
import urllib.request
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

import dblocks as db
from dblocks.training.optimizer import clip_grad_norm


def make_sample_text(size: int = 100_000) -> str:
    """Generate a simple repeating text for training."""
    base = (
        "The quick brown fox jumps over the lazy dog. "
        "A journey of a thousand miles begins with a single step. "
        "To be or not to be, that is the question. "
        "All that glitters is not gold. "
        "In the beginning was the word. "
        "Knowledge is power, and power corrupts. "
        "Time flies like an arrow, fruit flies like a banana. "
        "The only thing we have to fear is fear itself. "
    )
    repeats = (size // len(base)) + 1
    return (base * repeats)[:size]


def get_shakespeare_text(size_str: str) -> str:
    """Download and return the Tiny Shakespeare dataset of the specified size.

    If offline or download fails, falls back to synthetic text.
    """
    size_map = {
        "1k": 1000,
        "10k": 10000,
        "100k": 100000,
    }
    limit = size_map.get(size_str)

    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    cache_dir = os.path.join(os.path.dirname(__file__), "data")
    cache_path = os.path.join(cache_dir, "tinyshakespeare.txt")

    text = ""
    try:
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                text = f.read()
        else:
            print(f"  Downloading Tiny Shakespeare dataset from {url}...")
            os.makedirs(cache_dir, exist_ok=True)
            with urllib.request.urlopen(url, timeout=10) as response:
                content = response.read().decode("utf-8")
            with open(cache_path, "w", encoding="utf-8") as f:
                f.write(content)
            text = content
            print(f"  Saved to {cache_path}")
    except Exception as e:
        print(f"  Warning: Failed to load/download Tiny Shakespeare ({e}).")
        print("  Falling back to synthetic text generator.")
        fallback_size = limit if limit is not None else 100_000
        return make_sample_text(fallback_size)

    if limit is None:
        return text

    if len(text) < limit:
        repeats = (limit // max(1, len(text))) + 1
        text = (text * repeats)[:limit]
    else:
        text = text[:limit]

    return text


class StandardTrainer:
    """Standard end-to-end autoregressive trainer for fair comparison."""

    def __init__(
        self,
        model: db.DBlockGPT,
        lr: float = 3e-4,
        weight_decay: float = 0.1,
        warmup_steps: int = 100,
        max_steps: int = 0,
        seed: int = 42,
    ):
        self.model = model
        self.model.unfreeze()

        self.optimizer = optim.AdamW(
            learning_rate=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.95),
        )

        self.base_lr = lr
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self._step_count = 0
        mx.random.seed(seed)

        # Build the loss + grad function
        self._loss_and_grad_fn = nn.value_and_grad(
            model, self._compute_loss
        )

    def _get_lr(self) -> float:
        step = self._step_count
        if self.warmup_steps > 0 and step < self.warmup_steps:
            return self.base_lr * step / self.warmup_steps
        if self.max_steps > 0:
            progress = (step - self.warmup_steps) / max(
                1, self.max_steps - self.warmup_steps
            )
            progress = min(progress, 1.0)
            return self.base_lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))
        return self.base_lr

    def _compute_loss(
        self,
        model: db.DBlockGPT,
        tokens: mx.array,
    ) -> mx.array:
        logits = model(tokens)
        if logits.dtype != mx.float32:
            logits = logits.astype(mx.float32)
        shift_logits = logits[:, :-1, :]
        shift_targets = tokens[:, 1:]
        loss = mx.mean(nn.losses.cross_entropy(shift_logits.reshape(-1, logits.shape[-1]), shift_targets.reshape(-1)))
        return loss

    def train_step(self, tokens: mx.array) -> dict[str, Any]:
        self._step_count += 1
        lr = self._get_lr()
        self.optimizer.learning_rate = mx.array(lr)

        self.model.unfreeze()

        loss, grads = self._loss_and_grad_fn(self.model, tokens)

        # Clip gradients
        grads, grad_norm = clip_grad_norm(grads, 1.0)
        mx.eval(grad_norm)

        self.optimizer.update(self.model, grads)

        # Force evaluation
        mx.eval(self.model.parameters())
        mx.eval(self.optimizer.state)
        mx.eval(loss)

        return {
            "loss": float(loss),
            "lr": lr,
        }


def run_dblocks(model, batcher, args, total_steps, warmup_steps):
    print(f"\n[4/5] Training with DiffusionBlocks ({args.epochs} epochs × {args.steps} steps)...")
    print(f"  Warmup steps: {warmup_steps} ({args.warmup_pct * 100:.1f}% of {total_steps} total steps)")
    trainer = db.DBlockTrainer(
        model,
        lr=args.lr,
        warmup_steps=warmup_steps,
        max_steps=total_steps,
        lm_weight=1.0,
        callbacks=[
            db.ProgressCallback(log_every=max(1, args.steps // 10)),
            db.MemoryCallback(log_every=max(1, args.steps // 2)),
        ],
        seed=args.seed,
    )

    try:
        mx.reset_peak_memory()
    except Exception:
        try:
            mx.metal.reset_peak_memory()
        except Exception:
            pass

    start_time = time.time()
    history = trainer.fit(
        batcher, epochs=args.epochs, steps_per_epoch=args.steps,
    )
    train_time = time.time() - start_time

    peak_mem = 0.0
    try:
        peak_mem = mx.metal.get_peak_memory() / (1024 * 1024)
    except Exception:
        pass

    return history, train_time, peak_mem


def run_standard(model, batcher, args, total_steps, warmup_steps):
    print(f"\n[4/5] Training with Standard End-to-End ({args.epochs} epochs × {args.steps} steps)...")
    print(f"  Warmup steps: {warmup_steps} ({args.warmup_pct * 100:.1f}% of {total_steps} total steps)")
    trainer = StandardTrainer(
        model,
        lr=args.lr,
        warmup_steps=warmup_steps,
        max_steps=total_steps,
        seed=args.seed,
    )

    try:
        mx.reset_peak_memory()
    except Exception:
        try:
            mx.metal.reset_peak_memory()
        except Exception:
            pass

    history = []
    start_time = time.time()
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        epoch_losses = []
        step_in_epoch = 0
        epoch_start = time.time()

        for batch in batcher:
            global_step += 1
            step_in_epoch += 1

            if batch.dtype != mx.int32 and batch.dtype != mx.uint32:
                batch = batch.astype(mx.int32)

            res = trainer.train_step(batch)
            loss = res["loss"]
            lr = res["lr"]
            epoch_losses.append(loss)

            metrics = {
                "step": global_step,
                "epoch": epoch,
                "loss": loss,
                "lr": lr,
            }
            history.append(metrics)

            log_every = max(1, args.steps // 10)
            if global_step == 1 or global_step % log_every == 0:
                steps_per_sec = global_step / max(0.1, time.time() - start_time)
                print(
                    f"step {global_step:6d} | loss={loss:.4f} | lr={lr:.2e} | "
                    f"t={time.time() - start_time:.1f}s | {steps_per_sec:.1f} steps/s"
                )

            if step_in_epoch >= args.steps:
                break

        epoch_avg = sum(epoch_losses) / max(len(epoch_losses), 1)
        epoch_time = time.time() - epoch_start
        print(f"--- Epoch {epoch} complete | avg_loss={epoch_avg:.4f} | {epoch_time:.1f}s ---")

    train_time = time.time() - start_time

    peak_mem = 0.0
    try:
        peak_mem = mx.metal.get_peak_memory() / (1024 * 1024)
    except Exception:
        pass

    return history, train_time, peak_mem


def main():
    parser = argparse.ArgumentParser(description="Train DBlockGPT")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--steps", type=int, default=100, help="Steps per epoch")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=128, help="Sequence length")
    parser.add_argument("--dim", type=int, default=256, help="Model dimension")
    parser.add_argument("--depth", type=int, default=6, help="Number of layers")
    parser.add_argument("--heads", type=int, default=4, help="Attention heads")
    parser.add_argument("--num-blocks", type=int, default=2, help="DiffusionBlocks blocks")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument(
        "--warmup-pct",
        type=float,
        default=0.05,
        help="Warmup phase percentage of total training steps (e.g. 0.05 for 5%)",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="dblocks",
        choices=["dblocks", "standard", "both"],
        help="Training method to run",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--dataset",
        type=str,
        default="shakespeare",
        choices=["shakespeare", "synthetic"],
        help="Dataset to use for training",
    )
    parser.add_argument(
        "--data-size",
        type=str,
        default="100k",
        choices=["1k", "10k", "100k", "full"],
        help="Dataset size in tokens (characters)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("DiffusionBlocks vs Standard GPT Training Comparison")
    print("=" * 60)

    # 1. Create dataset
    print("\n[1/5] Creating dataset...")
    if args.dataset == "shakespeare":
        text = get_shakespeare_text(args.data_size)
        print(f"  Loaded Tiny Shakespeare ({args.data_size})")
    else:
        size_map = {"1k": 1000, "10k": 10000, "100k": 100000}
        size = size_map.get(args.data_size, 100_000)
        text = make_sample_text(size)
        print(f"  Generated Synthetic Text ({args.data_size})")

    if len(text) <= args.seq_len:
        print(f"  Error: Text length ({len(text)}) must be greater than sequence length ({args.seq_len})")
        return

    dataset, batcher = db.data.create_dataset(
        text, seq_len=args.seq_len, batch_size=args.batch_size,
    )
    print(f"  Text length: {len(text):,} chars")
    print(f"  Vocab size:  {dataset.vocab_size}")
    print(f"  Seq length:  {args.seq_len}")

    # 2. Build model config
    print("\n[2/5] Building model config...")
    config = db.GPTConfig(
        vocab_size=dataset.vocab_size,
        dim=args.dim,
        depth=args.depth,
        num_heads=args.heads,
        max_seq_len=args.seq_len,
        num_blocks=args.num_blocks,
        mlp_ratio=4.0,
        use_swiglu=True,
        dtype=mx.float16,
    )

    # We construct the model just to estimate parameters and memory
    temp_model = db.DBlockGPT(config)
    print(f"  Parameters:  {temp_model.num_parameters():,}")
    mem = temp_model.memory_estimate_mb()
    print(f"  Params:      {mem['params']:.1f} MB")
    print(f"  E2E peak:    {mem['e2e_peak_estimate']:.1f} MB")
    print(f"  DBlock peak: {mem['dblock_peak_estimate']:.1f} MB")
    savings = mem['e2e_peak_estimate'] / max(mem['dblock_peak_estimate'], 0.1)
    print(f"  Savings:     ~{savings:.1f}x memory reduction")

    # 3. Memory snapshot before training
    print("\n[3/5] Memory baseline...")
    print(f"  {db.memory_summary()}")

    total_steps = args.epochs * args.steps
    warmup_steps = int(args.warmup_pct * total_steps)

    history_db = None
    history_std = None
    time_db = 0.0
    time_std = 0.0
    peak_db = 0.0
    peak_std = 0.0

    # 4. Train with chosen method(s)
    if args.method in ["dblocks", "both"]:
        # Seed everything and build a fresh model for DiffusionBlocks
        mx.random.seed(args.seed)
        model = db.DBlockGPT(config)
        history_db, time_db, peak_db = run_dblocks(model, batcher, args, total_steps, warmup_steps)

        # 5. Summary
        print(f"\n[5/5] DiffusionBlocks Results")
        print(f"  Total time:  {time_db:.1f}s")
        print(f"  Total steps: {len(history_db)}")
        if history_db:
            final_loss = history_db[-1]["loss"]
            first_loss = history_db[0]["loss"]
            print(f"  First loss:  {first_loss:.4f}")
            print(f"  Final loss:  {final_loss:.4f}")
            print(f"  Reduction:   {first_loss - final_loss:.4f}")
            print(f"  Peak Memory: {peak_db:.1f} MB")

        # 6. Generate sample text
        print("\n[Bonus] Generating sample text with DiffusionBlocks model...")
        prompt = text[:20]
        prompt_tokens = mx.array([[dataset.char_to_idx.get(c, 0) for c in prompt]])
        model.eval()
        try:
            generated = model.generate(
                prompt_tokens, max_new_tokens=50, temperature=0.8,
            )
            output = dataset.decode(generated[0].tolist())
            print(f"  Prompt: '{prompt}'")
            print(f"  Output: '{output}'")
        except Exception as e:
            print(f"  Generation skipped ({e})")

    if args.method in ["standard", "both"]:
        # Seed everything and build a fresh model for Standard training
        mx.random.seed(args.seed)
        model = db.DBlockGPT(config)
        history_std, time_std, peak_std = run_standard(model, batcher, args, total_steps, warmup_steps)

        # 5. Summary
        print(f"\n[5/5] Standard E2E Results")
        print(f"  Total time:  {time_std:.1f}s")
        print(f"  Total steps: {len(history_std)}")
        if history_std:
            final_loss = history_std[-1]["loss"]
            first_loss = history_std[0]["loss"]
            print(f"  First loss:  {first_loss:.4f}")
            print(f"  Final loss:  {final_loss:.4f}")
            print(f"  Reduction:   {first_loss - final_loss:.4f}")
            print(f"  Peak Memory: {peak_std:.1f} MB")

        # 6. Generate sample text
        print("\n[Bonus] Generating sample text with Standard model...")
        prompt = text[:20]
        prompt_tokens = mx.array([[dataset.char_to_idx.get(c, 0) for c in prompt]])
        model.eval()
        try:
            generated = model.generate(
                prompt_tokens, max_new_tokens=50, temperature=0.8,
            )
            output = dataset.decode(generated[0].tolist())
            print(f"  Prompt: '{prompt}'")
            print(f"  Output: '{output}'")
        except Exception as e:
            print(f"  Generation skipped ({e})")

    # Side-by-side comparison summary and logging
    if args.method == "both":
        print("\n" + "=" * 60)
        print("Comparison Summary")
        print("=" * 60)
        print(f"{"Metrics":20} | {"DiffusionBlocks":16} | {"Standard E2E":16}")
        print(f"{"-"*20}-|-{"-"*16}-|-{"-"*16}")
        print(f"{"Final Loss":20} | {history_db[-1]['loss']:16.4f} | {history_std[-1]['loss']:16.4f}")
        print(f"{"Peak Memory (MB)":20} | {peak_db:16.1f} | {peak_std:16.1f}")
        print(f"{"Training Time (s)":20} | {time_db:16.1f} | {time_std:16.1f}")
        print(f"{"Steps / Second":20} | {len(history_db)/max(0.1, time_db):16.1f} | {len(history_std)/max(0.1, time_std):16.1f}")
        print("=" * 60)

        # Save comparative CSV
        csv_path = os.path.join(os.path.dirname(__file__), "dblocks_vs_standard.csv")
        try:
            import csv
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["step", "dblocks_loss", "dblocks_lr", "standard_loss", "standard_lr"])
                for i in range(total_steps):
                    db_step = history_db[i] if i < len(history_db) else {"loss": "", "lr": ""}
                    std_step = history_std[i] if i < len(history_std) else {"loss": "", "lr": ""}
                    writer.writerow([
                        i + 1,
                        db_step["loss"],
                        db_step["lr"],
                        std_step["loss"],
                        std_step["lr"]
                    ])
            print(f"\nSaved comparison results to {csv_path}")
        except Exception as e:
            print(f"Warning: Failed to save CSV ({e})")

    print("\nDone!")


if __name__ == "__main__":
    main()
