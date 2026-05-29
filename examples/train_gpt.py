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
import os
import time
import urllib.request

import mlx.core as mx

import dblocks as db


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
    print("DiffusionBlocks GPT Training Example")
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

    # 2. Build model
    print("\n[2/5] Building model...")
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
    model = db.DBlockGPT(config)

    print(f"  Parameters:  {model.num_parameters():,}")
    mem = model.memory_estimate_mb()
    print(f"  Params:      {mem['params']:.1f} MB")
    print(f"  E2E peak:    {mem['e2e_peak_estimate']:.1f} MB")
    print(f"  DBlock peak: {mem['dblock_peak_estimate']:.1f} MB")
    savings = mem['e2e_peak_estimate'] / max(mem['dblock_peak_estimate'], 0.1)
    print(f"  Savings:     ~{savings:.1f}x memory reduction")

    # Show block specs
    print(f"\n  Block assignments:")
    for spec in model.block_specs:
        print(
            f"    Block {spec.model_index}: layers {spec.layers}, "
            f"σ=[{spec.sigma_min:.4f}, {spec.sigma_max:.4f}]"
        )

    # 3. Memory snapshot before training
    print("\n[3/5] Memory baseline...")
    print(f"  {db.memory_summary()}")

    # 4. Train
    print(f"\n[4/5] Training ({args.epochs} epochs × {args.steps} steps)...")
    trainer = db.DBlockTrainer(
        model,
        lr=args.lr,
        warmup_steps=min(50, args.steps),
        max_steps=args.epochs * args.steps,
        lm_weight=1.0,
        callbacks=[
            db.ProgressCallback(log_every=10),
            db.MemoryCallback(log_every=50),
        ],
        seed=args.seed,
    )

    start_time = time.time()
    history = trainer.fit(
        batcher, epochs=args.epochs, steps_per_epoch=args.steps,
    )
    train_time = time.time() - start_time

    # 5. Summary
    print(f"\n[5/5] Results")
    print(f"  Total time:  {train_time:.1f}s")
    print(f"  Total steps: {len(history)}")
    if history:
        final_loss = history[-1]["loss"]
        first_loss = history[0]["loss"]
        print(f"  First loss:  {first_loss:.4f}")
        print(f"  Final loss:  {final_loss:.4f}")
        print(f"  Reduction:   {first_loss - final_loss:.4f}")

    print(f"\n  {db.memory_summary()}")

    # 6. Generate sample text
    print("\n[Bonus] Generating sample text...")
    prompt = text[:20]
    prompt_tokens = mx.array([[dataset.char_to_idx.get(c, 0) for c in prompt]])

    model.eval()  # Disable dropout
    try:
        generated = model.generate(
            prompt_tokens, max_new_tokens=50, temperature=0.8,
        )
        output = dataset.decode(generated[0].tolist())
        print(f"  Prompt: '{prompt}'")
        print(f"  Output: '{output}'")
    except Exception as e:
        print(f"  Generation skipped ({e})")

    print("\nDone!")


if __name__ == "__main__":
    main()
