"""
GradTracer Visualization Module

Provides human-readable, visual diagnostics for embeddings, trees, and dense layers.
"""
from typing import Optional, List

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_embedding_diagnostics(tracker, top_k: int = 20, save_path: Optional[str] = None):
    """
    Plots a human-readable visual diagnostic for an EmbeddingTracker.
    
    Creates a 1x3 panel figure showing:
    1. Exposure Frequency (Popularity Bias)
    2. Embedding Velocity (Zombie vs Healthy)
    3. Oscillation Scores
    """
    # Defensive check: Ensure there's data to plot
    if tracker.steps == 0 or np.sum(tracker.freqs) == 0:
        print("⚠️ No data tracked yet. Run training steps with the tracker active before plotting.")
        return

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Identify specific sets
    dead_idx = tracker.dead_embeddings()
    zombie_idx = tracker.zombie_embeddings()
    
    # Sort by frequency (Popularity)
    sorted_idx = np.argsort(tracker.freqs)[::-1]
    plot_idx = sorted_idx[:top_k]
    
    # 1. Frequency Distribution Bar Plot
    freq_vals = tracker.freqs[plot_idx]
    
    colors1 = []
    for idx in plot_idx:
        if idx in dead_idx: colors1.append("gray")
        elif idx in zombie_idx: colors1.append("red")
        else: colors1.append("steelblue")
        
    axes[0].bar(range(len(plot_idx)), freq_vals, color=colors1)
    axes[0].set_title(f"Top {top_k} Exposed Embeddings", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Embedding Rank (by popularity)")
    axes[0].set_ylabel("Update Frequency")
    axes[0].set_xticks(range(len(plot_idx)))
    axes[0].set_xticklabels([f"ID:{i}" for i in plot_idx], rotation=90)
    
    # 2. Velocity Space (Scatter)
    active_mask = tracker.freqs > 0
    active_vels = tracker.velocities[active_mask]
    active_freqs = tracker.freqs[active_mask]
    active_oscil = tracker.oscillation_scores[active_mask]
    
    axes[1].scatter(active_freqs, active_vels, alpha=0.4, color="steelblue", label="Healthy")
    
    if len(zombie_idx) > 0:
        z_freqs = tracker.freqs[zombie_idx]
        z_vels = tracker.velocities[zombie_idx]
        axes[1].scatter(z_freqs, z_vels, color="red", alpha=0.9, edgecolor="black", label="Zombies (Oscillating)")
        
    axes[1].set_title("Velocity vs Exposure", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Exposure Frequency")
    axes[1].set_ylabel("EMA Velocity (Change Magnitude)")
    axes[1].legend()
    
    # 3. Oscillation Distribution
    valid_oscil = tracker.oscillation_scores[tracker.freqs > 5]
    if len(valid_oscil) > 0:
        sns.histplot(valid_oscil, bins=30, ax=axes[2], color="purple", kde=True)
        axes[2].axvline(x=-0.3, color="red", linestyle="--", label="Zombie Threshold (-0.3)")
    
    axes[2].set_title("Oscillation Distribution (Cos Sim)", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("EMA Cosine Similarity to Prev Step")
    axes[2].set_ylabel("Count")
    axes[2].legend()
    
    plt.tight_layout()
    
    # Add an explanatory text box at the bottom
    caption = (
        "🧠 HUMAN-READABLE DIAGNOSTIC GUIDE:\n"
        "• Blue (Healthy): These embeddings are learning efficiently. They move when updated.\n"
        "• Red (Zombies): High update frequency, but direction constantly reverses (Oscillation < -0.3). They are stuck in a tug-of-war. Action: Use SparseAdam or reduce LR.\n"
        "• Gray/Zero (Dead): Never updated. Check dataloader negative sampling. Action: Downsample or Hash."
    )
    plt.figtext(0.5, -0.15, caption, wrap=True, horizontalalignment='center', fontsize=11, 
                bbox={"facecolor":"#f9f9f9", "alpha":0.8, "pad":10, "boxstyle":"round,pad=1"})
    
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"📊 Saved visualization to {save_path}")
    else:
        plt.show()
