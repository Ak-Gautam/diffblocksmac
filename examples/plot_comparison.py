#!/usr/bin/env python3
"""Plot comparison between DiffusionBlocks and Standard training."""

import os
import matplotlib.pyplot as plt
import pandas as pd

def main():
    csv_path = os.path.join(os.path.dirname(__file__), "dblocks_vs_standard.csv")
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} does not exist. Run the training script first with '--method both'.")
        return

    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Create figures path
    fig_path = os.path.join(os.path.dirname(__file__), "dblocks_vs_standard.png")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 1. Loss curves plot
    axes[0].plot(df["step"], df["dblocks_loss"], label="DiffusionBlocks Loss", color="#1f77b4", linewidth=2)
    axes[0].plot(df["step"], df["standard_loss"], label="Standard E2E Loss", color="#ff7f0e", linewidth=2)
    axes[0].set_title("Training Loss Comparison", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Steps", fontsize=12)
    axes[0].set_ylabel("Loss", fontsize=12)
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(fontsize=11)

    # 2. Learning Rate schedule plot
    axes[1].plot(df["step"], df["dblocks_lr"], label="DiffusionBlocks LR", color="#2ca02c", linewidth=2)
    axes[1].plot(df["step"], df["standard_lr"], label="Standard E2E LR", color="#d62728", linestyle="--", linewidth=2)
    axes[1].set_title("Learning Rate Schedule Comparison", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Steps", fontsize=12)
    axes[1].set_ylabel("Learning Rate", fontsize=12)
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    print(f"Saved comparison plot to {fig_path}")

if __name__ == "__main__":
    main()
