"""
Visualization suite for GradTracer.

Provides DLPlotAPI (for PyTorch models) and BoostingPlotAPI (for tree models).
All plots return matplotlib Figure objects for easy customization and saving.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.figure import Figure

from gradtracer.snapshot import BoostingStore, SnapshotStore


# ──────────────────────────────────────────────────────────────────────
#  Color palette
# ──────────────────────────────────────────────────────────────────────
PALETTE = {
    "primary": "#6366F1",     # indigo
    "secondary": "#EC4899",   # pink
    "success": "#10B981",     # emerald
    "warning": "#F59E0B",     # amber
    "danger": "#EF4444",      # red
    "info": "#3B82F6",        # blue
    "bg": "#1E1E2E",          # dark bg
    "text": "#CDD6F4",        # light text
    "grid": "#45475A",        # grid lines
}


def _apply_style(fig: Figure, ax_or_axes):
    """Apply dark theme to figure."""
    fig.patch.set_facecolor(PALETTE["bg"])
    axes = ax_or_axes if hasattr(ax_or_axes, "__iter__") else [ax_or_axes]
    for ax in (a for row in axes for a in (row if hasattr(row, "__iter__") else [row])):
        ax.set_facecolor(PALETTE["bg"])
        ax.tick_params(colors=PALETTE["text"], labelsize=8)
        ax.xaxis.label.set_color(PALETTE["text"])
        ax.yaxis.label.set_color(PALETTE["text"])
        ax.title.set_color(PALETTE["text"])
        for spine in ax.spines.values():
            spine.set_color(PALETTE["grid"])
        ax.grid(True, color=PALETTE["grid"], alpha=0.3, linewidth=0.5)


def _short_name(name: str) -> str:
    """Shorten 'layer3.conv.weight' → 'L3.conv.W'"""
    parts = name.replace(".weight", ".W").replace(".bias", ".b").split(".")
    shortened = []
    for p in parts:
        if p.startswith("layer") or p.startswith("features"):
            shortened.append("L" + p.replace("layer", "").replace("features", ""))
        else:
            shortened.append(p[:8])
    return ".".join(shortened)


# ======================================================================
#  DL Plot API
# ======================================================================

class DLPlotAPI:
    """Visualization for DL (PyTorch) training dynamics."""

    def __init__(self, store: SnapshotStore):
        self.store = store

    def loss(self, figsize: Tuple[int, int] = (10, 4), **kwargs) -> Figure:
        """Plot loss curve."""
        losses = self.store.get_loss_history()
        steps = list(range(1, len(losses) + 1))

        fig, ax = plt.subplots(figsize=figsize)
        _apply_style(fig, ax)

        valid = [(s, l) for s, l in zip(steps, losses) if l is not None]
        if valid:
            ss, ll = zip(*valid)
            ax.plot(ss, ll, color=PALETTE["primary"], linewidth=2, label="Loss")
            ax.scatter([ss[0], ss[-1]], [ll[0], ll[-1]],
                       color=PALETTE["secondary"], zorder=5, s=40)
            ax.set_xlabel("Step")
            ax.set_ylabel("Loss")
            ax.set_title("Training Loss")
            ax.legend(facecolor=PALETTE["bg"], edgecolor=PALETTE["grid"],
                      labelcolor=PALETTE["text"])

        fig.tight_layout()
        return fig

    def velocity_heatmap(self, figsize: Tuple[int, int] = (12, 6), **kwargs) -> Figure:
        """
        Layer × Step heatmap showing weight change velocity.
        Hot = fast learning, Cold = stagnant.
        """
        from gradtracer.analyzers.velocity import velocity_heatmap_data
        data, layer_names, steps = velocity_heatmap_data(self.store)

        short_names = [_short_name(n) for n in layer_names]

        fig, ax = plt.subplots(figsize=figsize)
        _apply_style(fig, ax)

        # Log-scale for better visibility
        log_data = np.log10(data + 1e-12)
        im = ax.imshow(log_data, aspect="auto", cmap="inferno",
                       interpolation="nearest")
        cbar = fig.colorbar(im, ax=ax, label="log₁₀(velocity)")
        cbar.ax.yaxis.label.set_color(PALETTE["text"])
        cbar.ax.tick_params(colors=PALETTE["text"])

        ax.set_yticks(range(len(short_names)))
        ax.set_yticklabels(short_names, fontsize=7)
        ax.set_xlabel("Step")
        ax.set_ylabel("Layer")
        ax.set_title("Weight Velocity Heatmap (log scale)")

        fig.tight_layout()
        return fig

    def gradient_flow(self, figsize: Tuple[int, int] = (12, 5), **kwargs) -> Figure:
        """Plot gradient norm per layer at the latest step."""
        if self.store.num_steps == 0:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "No data yet", ha="center", va="center")
            return fig

        latest = self.store.steps[-1]
        names = list(latest.layers.keys())
        grad_norms = [latest.layers[n].grad_norm for n in names]
        short_names = [_short_name(n) for n in names]

        fig, ax = plt.subplots(figsize=figsize)
        _apply_style(fig, ax)

        colors = []
        for gn in grad_norms:
            if gn > 10:
                colors.append(PALETTE["danger"])
            elif gn < 1e-5:
                colors.append(PALETTE["warning"])
            else:
                colors.append(PALETTE["success"])

        ax.barh(range(len(short_names)), grad_norms, color=colors, alpha=0.85)
        ax.set_yticks(range(len(short_names)))
        ax.set_yticklabels(short_names, fontsize=7)
        ax.set_xlabel("Gradient Norm")
        ax.set_title(f"Gradient Flow (Step {self.store.num_steps})")
        ax.set_xscale("log")

        fig.tight_layout()
        return fig

    def weight_distribution(
        self, layers: Optional[List[str]] = None, figsize: Tuple[int, int] = (12, 6),
        **kwargs
    ) -> Figure:
        """Plot weight norm distribution evolution for selected layers."""
        if layers is None:
            layers = self.store.layer_names[:6]  # top 6

        n = len(layers)
        fig, axes = plt.subplots(1, n, figsize=figsize, squeeze=False)
        _apply_style(fig, axes)

        for idx, name in enumerate(layers):
            ax = axes[0][idx]
            norms = self.store.get_layer_series(name, "weight_norm")
            means = self.store.get_layer_series(name, "weight_mean")
            stds = self.store.get_layer_series(name, "weight_std")
            steps = list(range(1, len(norms) + 1))

            ax.plot(steps, norms, color=PALETTE["primary"], linewidth=1.5, label="norm")
            ax.fill_between(
                steps,
                [m - s for m, s in zip(means, stds)],
                [m + s for m, s in zip(means, stds)],
                alpha=0.2, color=PALETTE["info"],
            )
            ax.set_title(_short_name(name), fontsize=8)
            ax.set_xlabel("Step", fontsize=7)

        fig.suptitle("Weight Distribution Evolution", color=PALETTE["text"], fontsize=12)
        fig.tight_layout()
        return fig

    def health_dashboard(self, figsize: Tuple[int, int] = (12, 5), **kwargs) -> Figure:
        """Bar chart of per-layer health scores (0-100)."""
        from gradtracer.analyzers.health import layer_health_score

        scores = layer_health_score(self.store)
        names = list(scores.keys())
        vals = list(scores.values())
        short_names = [_short_name(n) for n in names]

        fig, ax = plt.subplots(figsize=figsize)
        _apply_style(fig, ax)

        colors = []
        for v in vals:
            if v >= 70:
                colors.append(PALETTE["success"])
            elif v >= 40:
                colors.append(PALETTE["warning"])
            else:
                colors.append(PALETTE["danger"])

        ax.barh(range(len(short_names)), vals, color=colors, alpha=0.85)
        ax.set_yticks(range(len(short_names)))
        ax.set_yticklabels(short_names, fontsize=7)
        ax.set_xlabel("Health Score")
        ax.set_title("Layer Health Dashboard")
        ax.set_xlim(0, 100)

        # Add score text
        for i, v in enumerate(vals):
            ax.text(v + 1, i, f"{v:.0f}", va="center", fontsize=7,
                    color=PALETTE["text"])

        fig.tight_layout()
        return fig

    def gradient_snr(self, figsize: Tuple[int, int] = (12, 5), **kwargs) -> Figure:
        """Plot gradient SNR per layer over time."""
        from gradtracer.analyzers.health import gradient_snr_per_layer

        snr_data = gradient_snr_per_layer(self.store)

        fig, ax = plt.subplots(figsize=figsize)
        _apply_style(fig, ax)

        cmap = plt.cm.get_cmap("cool", len(snr_data))
        for idx, (name, series) in enumerate(snr_data.items()):
            clipped = [min(s, 1e6) for s in series]
            steps = list(range(1, len(clipped) + 1))
            ax.plot(steps, clipped, linewidth=1.2, alpha=0.7,
                    color=cmap(idx), label=_short_name(name))

        ax.set_xlabel("Step")
        ax.set_ylabel("Gradient SNR")
        ax.set_title("Gradient Signal-to-Noise Ratio")
        ax.set_yscale("log")
        if len(snr_data) <= 10:
            ax.legend(fontsize=6, facecolor=PALETTE["bg"],
                      edgecolor=PALETTE["grid"], labelcolor=PALETTE["text"])

        fig.tight_layout()
        return fig

    def full_report(self, figsize: Tuple[int, int] = (16, 14), **kwargs) -> Figure:
        """All-in-one dashboard with 6 subplots."""
        fig = plt.figure(figsize=figsize)
        fig.patch.set_facecolor(PALETTE["bg"])
        fig.suptitle("GradTracer Training Report", fontsize=16,
                     color=PALETTE["text"], fontweight="bold")

        gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)

        # 1) Loss curve
        ax1 = fig.add_subplot(gs[0, 0])
        losses = self.store.get_loss_history()
        valid = [(i + 1, l) for i, l in enumerate(losses) if l is not None]
        if valid:
            ss, ll = zip(*valid)
            ax1.plot(ss, ll, color=PALETTE["primary"], linewidth=2)
        ax1.set_title("Loss Curve", fontsize=10)
        ax1.set_xlabel("Step", fontsize=8)

        # 2) Velocity heatmap
        ax2 = fig.add_subplot(gs[0, 1])
        from gradtracer.analyzers.velocity import velocity_heatmap_data
        data, layer_names, steps = velocity_heatmap_data(self.store)
        log_data = np.log10(data + 1e-12)
        ax2.imshow(log_data, aspect="auto", cmap="inferno", interpolation="nearest")
        ax2.set_title("Velocity Heatmap", fontsize=10)
        ax2.set_yticks(range(len(layer_names)))
        ax2.set_yticklabels([_short_name(n) for n in layer_names], fontsize=5)

        # 3) Gradient flow
        ax3 = fig.add_subplot(gs[1, 0])
        if self.store.num_steps > 0:
            latest = self.store.steps[-1]
            names = list(latest.layers.keys())
            gnorms = [latest.layers[n].grad_norm for n in names]
            colors = [PALETTE["danger"] if g > 10 else PALETTE["success"] for g in gnorms]
            ax3.barh(range(len(names)), gnorms, color=colors, alpha=0.85)
            ax3.set_yticks(range(len(names)))
            ax3.set_yticklabels([_short_name(n) for n in names], fontsize=5)
            ax3.set_xscale("log")
        ax3.set_title("Gradient Flow", fontsize=10)

        # 4) Health scores
        ax4 = fig.add_subplot(gs[1, 1])
        from gradtracer.analyzers.health import layer_health_score
        scores = layer_health_score(self.store)
        s_names = list(scores.keys())
        s_vals = list(scores.values())
        h_colors = [PALETTE["success"] if v >= 70 else PALETTE["warning"] if v >= 40
                     else PALETTE["danger"] for v in s_vals]
        ax4.barh(range(len(s_names)), s_vals, color=h_colors, alpha=0.85)
        ax4.set_yticks(range(len(s_names)))
        ax4.set_yticklabels([_short_name(n) for n in s_names], fontsize=5)
        ax4.set_xlim(0, 100)
        ax4.set_title("Health Scores", fontsize=10)

        # 5) Gradient SNR
        ax5 = fig.add_subplot(gs[2, 0])
        from gradtracer.analyzers.health import gradient_snr_per_layer
        snr_data = gradient_snr_per_layer(self.store)
        cmap = plt.cm.get_cmap("cool", max(len(snr_data), 1))
        for idx, (name, series) in enumerate(snr_data.items()):
            clipped = [min(s, 1e6) for s in series]
            ax5.plot(range(1, len(clipped) + 1), clipped, linewidth=1,
                     alpha=0.7, color=cmap(idx))
        ax5.set_yscale("log")
        ax5.set_title("Gradient SNR", fontsize=10)
        ax5.set_xlabel("Step", fontsize=8)

        # 6) Velocity trends (top layers by latest velocity)
        ax6 = fig.add_subplot(gs[2, 1])
        from gradtracer.analyzers.velocity import velocity_per_layer
        vel_data = velocity_per_layer(self.store)
        # Sort by latest velocity, show top 5
        sorted_layers = sorted(
            vel_data.items(),
            key=lambda x: x[1][-1] if x[1] else 0,
            reverse=True
        )[:5]
        for name, series in sorted_layers:
            ax6.plot(range(1, len(series) + 1), series, linewidth=1.2,
                     label=_short_name(name))
        ax6.set_title("Top-5 Layer Velocities", fontsize=10)
        ax6.set_xlabel("Step", fontsize=8)
        ax6.legend(fontsize=6, facecolor=PALETTE["bg"],
                   edgecolor=PALETTE["grid"], labelcolor=PALETTE["text"])

        # Apply style to all axes
        for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
            ax.set_facecolor(PALETTE["bg"])
            ax.tick_params(colors=PALETTE["text"], labelsize=6)
            ax.xaxis.label.set_color(PALETTE["text"])
            ax.yaxis.label.set_color(PALETTE["text"])
            ax.title.set_color(PALETTE["text"])
            for spine in ax.spines.values():
                spine.set_color(PALETTE["grid"])

        return fig


# ======================================================================
#  Boosting Plot API
# ======================================================================

class BoostingPlotAPI:
    """Visualization for boosting model (XGBoost / LightGBM / CatBoost) training."""

    def __init__(self, store: BoostingStore):
        self.store = store

    def eval_metrics(self, figsize: Tuple[int, int] = (10, 5), **kwargs) -> Figure:
        """Plot evaluation metrics across rounds."""
        fig, ax = plt.subplots(figsize=figsize)
        _apply_style(fig, ax)

        colors = [PALETTE["primary"], PALETTE["secondary"], PALETTE["success"],
                  PALETTE["info"], PALETTE["warning"]]
        color_idx = 0

        for ds in self.store.get_all_dataset_names():
            for metric in self.store.get_all_metric_names():
                series = self.store.get_eval_metric_series(ds, metric)
                rounds = list(range(1, len(series) + 1))
                ax.plot(rounds, series, linewidth=1.5,
                        color=colors[color_idx % len(colors)],
                        label=f"{ds}/{metric}")
                color_idx += 1

        ax.set_xlabel("Round")
        ax.set_ylabel("Metric Value")
        ax.set_title("Evaluation Metrics")
        ax.legend(facecolor=PALETTE["bg"], edgecolor=PALETTE["grid"],
                  labelcolor=PALETTE["text"])

        fig.tight_layout()
        return fig

    def feature_drift(
        self, top_k: int = 10, figsize: Tuple[int, int] = (12, 6), **kwargs
    ) -> Figure:
        """
        Show how feature importance changes across boosting rounds.

        Highlights features that gained or lost importance over training.
        """
        features = self.store.get_all_feature_names()
        if not features:
            fig, ax = plt.subplots(figsize=figsize)
            _apply_style(fig, ax)
            ax.text(0.5, 0.5, "No feature importance data available",
                    ha="center", va="center", color=PALETTE["text"])
            return fig

        # Compute total importance to select top-k
        totals = {}
        for feat in features:
            series = self.store.get_feature_importance_series(feat)
            totals[feat] = sum(series)

        top_features = sorted(totals.keys(), key=lambda f: totals[f], reverse=True)[:top_k]

        fig, ax = plt.subplots(figsize=figsize)
        _apply_style(fig, ax)

        cmap = plt.cm.get_cmap("tab10", len(top_features))
        for idx, feat in enumerate(top_features):
            series = self.store.get_feature_importance_series(feat)
            rounds = list(range(1, len(series) + 1))
            ax.plot(rounds, series, linewidth=1.5, color=cmap(idx), label=feat, alpha=0.85)

        ax.set_xlabel("Round")
        ax.set_ylabel("Feature Importance (gain)")
        ax.set_title(f"Feature Importance Drift (Top {top_k})")
        ax.legend(fontsize=7, facecolor=PALETTE["bg"],
                  edgecolor=PALETTE["grid"], labelcolor=PALETTE["text"])

        fig.tight_layout()
        return fig

    def feature_importance_heatmap(
        self, top_k: int = 15, figsize: Tuple[int, int] = (14, 7), **kwargs
    ) -> Figure:
        """Feature × Round heatmap for importance evolution."""
        features = self.store.get_all_feature_names()
        if not features:
            fig, ax = plt.subplots(figsize=figsize)
            _apply_style(fig, ax)
            ax.text(0.5, 0.5, "No feature importance data", ha="center",
                    va="center", color=PALETTE["text"])
            return fig

        # Select top-k by total importance
        totals = {f: sum(self.store.get_feature_importance_series(f)) for f in features}
        top_features = sorted(totals, key=lambda f: totals[f], reverse=True)[:top_k]

        n_rounds = self.store.num_rounds
        data = np.zeros((len(top_features), n_rounds))
        for i, feat in enumerate(top_features):
            series = self.store.get_feature_importance_series(feat)
            data[i, :len(series)] = series

        fig, ax = plt.subplots(figsize=figsize)
        _apply_style(fig, ax)

        im = ax.imshow(data, aspect="auto", cmap="YlOrRd", interpolation="nearest")
        cbar = fig.colorbar(im, ax=ax, label="Importance")
        cbar.ax.yaxis.label.set_color(PALETTE["text"])
        cbar.ax.tick_params(colors=PALETTE["text"])

        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features, fontsize=7)
        ax.set_xlabel("Round")
        ax.set_ylabel("Feature")
        ax.set_title(f"Feature Importance Heatmap (Top {top_k})")

        fig.tight_layout()
        return fig

    def overfitting_detector(self, figsize: Tuple[int, int] = (10, 5), **kwargs) -> Figure:
        """
        Plot train vs validation metrics to highlight overfitting regions.
        Shades the area where validation diverges from training.
        """
        datasets = self.store.get_all_dataset_names()
        metrics = self.store.get_all_metric_names()

        if not datasets or not metrics:
            fig, ax = plt.subplots(figsize=figsize)
            _apply_style(fig, ax)
            ax.text(0.5, 0.5, "No eval metric data", ha="center",
                    va="center", color=PALETTE["text"])
            return fig

        fig, ax = plt.subplots(figsize=figsize)
        _apply_style(fig, ax)

        metric = metrics[0]  # Use first metric
        train_ds = None
        valid_ds = None
        for ds in datasets:
            ds_lower = ds.lower()
            if "train" in ds_lower or "learn" in ds_lower:
                train_ds = ds
            elif "valid" in ds_lower or "test" in ds_lower or "eval" in ds_lower:
                valid_ds = ds

        if not train_ds:
            train_ds = datasets[0]
        if not valid_ds and len(datasets) > 1:
            valid_ds = datasets[1]

        train_series = self.store.get_eval_metric_series(train_ds, metric)
        rounds = list(range(1, len(train_series) + 1))
        ax.plot(rounds, train_series, color=PALETTE["primary"],
                linewidth=2, label=f"Train ({metric})")

        if valid_ds:
            valid_series = self.store.get_eval_metric_series(valid_ds, metric)
            ax.plot(rounds[:len(valid_series)], valid_series,
                    color=PALETTE["secondary"], linewidth=2,
                    label=f"Valid ({metric})")

            # Shade overfitting region
            min_len = min(len(train_series), len(valid_series))
            for i in range(1, min_len):
                gap = abs(valid_series[i] - train_series[i])
                prev_gap = abs(valid_series[i - 1] - train_series[i - 1])
                if gap > prev_gap * 1.5 and gap > 0.01:
                    ax.axvspan(i + 1, min_len, alpha=0.1, color=PALETTE["danger"])
                    ax.annotate("⚠ Overfitting zone", xy=(i + 1, train_series[i]),
                                fontsize=8, color=PALETTE["danger"])
                    break

        ax.set_xlabel("Round")
        ax.set_ylabel(metric)
        ax.set_title("Overfitting Detector")
        ax.legend(facecolor=PALETTE["bg"], edgecolor=PALETTE["grid"],
                  labelcolor=PALETTE["text"])

        fig.tight_layout()
        return fig

    def full_report(self, top_k: int = 10, figsize: Tuple[int, int] = (16, 10), **kwargs) -> Figure:
        """All-in-one boosting dashboard."""
        fig = plt.figure(figsize=figsize)
        fig.patch.set_facecolor(PALETTE["bg"])
        fig.suptitle("GradTracer Boosting Report", fontsize=16,
                     color=PALETTE["text"], fontweight="bold")

        gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

        # 1) Eval metrics
        ax1 = fig.add_subplot(gs[0, 0])
        colors = [PALETTE["primary"], PALETTE["secondary"], PALETTE["success"]]
        c_idx = 0
        for ds in self.store.get_all_dataset_names():
            for metric in self.store.get_all_metric_names():
                series = self.store.get_eval_metric_series(ds, metric)
                ax1.plot(range(1, len(series) + 1), series, linewidth=1.5,
                         color=colors[c_idx % len(colors)], label=f"{ds}/{metric}")
                c_idx += 1
        ax1.set_title("Eval Metrics", fontsize=10)
        ax1.legend(fontsize=6, facecolor=PALETTE["bg"],
                   edgecolor=PALETTE["grid"], labelcolor=PALETTE["text"])

        # 2) Feature drift
        ax2 = fig.add_subplot(gs[0, 1])
        features = self.store.get_all_feature_names()
        if features:
            totals = {f: sum(self.store.get_feature_importance_series(f)) for f in features}
            top = sorted(totals, key=lambda f: totals[f], reverse=True)[:5]
            cmap = plt.cm.get_cmap("tab10", len(top))
            for i, feat in enumerate(top):
                s = self.store.get_feature_importance_series(feat)
                ax2.plot(range(1, len(s) + 1), s, linewidth=1.2,
                         color=cmap(i), label=feat)
            ax2.legend(fontsize=5, facecolor=PALETTE["bg"],
                       edgecolor=PALETTE["grid"], labelcolor=PALETTE["text"])
        ax2.set_title("Feature Drift (Top 5)", fontsize=10)

        # 3) Feature importance heatmap
        ax3 = fig.add_subplot(gs[1, 0])
        if features:
            totals = {f: sum(self.store.get_feature_importance_series(f)) for f in features}
            top = sorted(totals, key=lambda f: totals[f], reverse=True)[:10]
            data = np.zeros((len(top), self.store.num_rounds))
            for i, feat in enumerate(top):
                s = self.store.get_feature_importance_series(feat)
                data[i, :len(s)] = s
            ax3.imshow(data, aspect="auto", cmap="YlOrRd", interpolation="nearest")
            ax3.set_yticks(range(len(top)))
            ax3.set_yticklabels(top, fontsize=5)
        ax3.set_title("Importance Heatmap", fontsize=10)

        # 4) Overfitting check
        ax4 = fig.add_subplot(gs[1, 1])
        datasets = self.store.get_all_dataset_names()
        metrics_list = self.store.get_all_metric_names()
        if datasets and metrics_list:
            met = metrics_list[0]
            for ds in datasets[:2]:
                s = self.store.get_eval_metric_series(ds, met)
                ax4.plot(range(1, len(s) + 1), s, linewidth=1.5, label=ds)
            ax4.legend(fontsize=6, facecolor=PALETTE["bg"],
                       edgecolor=PALETTE["grid"], labelcolor=PALETTE["text"])
        ax4.set_title("Train vs Valid", fontsize=10)

        for ax in [ax1, ax2, ax3, ax4]:
            ax.set_facecolor(PALETTE["bg"])
            ax.tick_params(colors=PALETTE["text"], labelsize=6)
            ax.xaxis.label.set_color(PALETTE["text"])
            ax.yaxis.label.set_color(PALETTE["text"])
            ax.title.set_color(PALETTE["text"])
            for spine in ax.spines.values():
                spine.set_color(PALETTE["grid"])

        return fig


# ======================================================================
#  Compression Plot API
# ======================================================================

class CompressionPlotAPI:
    """Visualization for model compression analysis."""

    def __init__(self, tracker):
        self.tracker = tracker

    def tradeoff_curve(self, figsize=(10, 6)) -> Figure:
        """
        Pareto frontier: model size vs performance.

        Each snapshot is a point; the Pareto front connects non-dominated points.
        Highlights the optimal compression point.
        """
        snaps = self.tracker.snapshots
        if not snaps:
            fig, ax = plt.subplots(figsize=figsize)
            _apply_style(fig, ax)
            ax.text(0.5, 0.5, "No snapshots recorded", ha="center", va="center",
                    color=PALETTE["text"])
            return fig

        fig, ax = plt.subplots(figsize=figsize)
        _apply_style(fig, ax)

        sizes = [s.model_size_mb for s in snaps]
        scores = [s.eval_metrics.get("score", 0) for s in snaps]
        names = [s.name for s in snaps]
        sparsities = [s.sparsity for s in snaps]

        # Color by sparsity
        scatter = ax.scatter(sizes, scores, c=sparsities, cmap="RdYlGn_r",
                            s=100, zorder=5, edgecolors="white", linewidth=0.5)
        cbar = fig.colorbar(scatter, ax=ax, label="Sparsity")
        cbar.ax.yaxis.label.set_color(PALETTE["text"])
        cbar.ax.tick_params(colors=PALETTE["text"])

        # Annotate points
        for i, name in enumerate(names):
            ax.annotate(name, (sizes[i], scores[i]), textcoords="offset points",
                       xytext=(5, 5), fontsize=6, color=PALETTE["text"], alpha=0.8)

        # Connect with line (sorted by size)
        sorted_idx = sorted(range(len(sizes)), key=lambda i: sizes[i])
        ax.plot([sizes[i] for i in sorted_idx], [scores[i] for i in sorted_idx],
                color=PALETTE["info"], alpha=0.4, linewidth=1, linestyle="--")

        # Mark original and best
        if snaps:
            ax.scatter([sizes[0]], [scores[0]], s=200, marker="*",
                      color=PALETTE["warning"], zorder=6, label="Original")

            # Best = highest sparsity with good score
            orig_score = scores[0]
            valid = [(i, s) for i, s in enumerate(snaps) if s.eval_metrics.get("score", 0) >= orig_score * 0.95]
            if valid and len(valid) > 1:
                best_idx = max(valid, key=lambda x: x[1].sparsity)[0]
                ax.scatter([sizes[best_idx]], [scores[best_idx]], s=200, marker="D",
                          color=PALETTE["success"], zorder=6, label="Optimal")

        # Performance floor line
        if scores:
            floor = scores[0] * 0.95
            ax.axhline(y=floor, color=PALETTE["danger"], linestyle=":", alpha=0.7,
                      label="95% floor")

        ax.set_xlabel("Model Size (MB)")
        ax.set_ylabel("Performance Score")
        ax.set_title("Compression Tradeoff Curve")
        ax.legend(facecolor=PALETTE["bg"], edgecolor=PALETTE["grid"],
                 labelcolor=PALETTE["text"], fontsize=8)

        fig.tight_layout()
        return fig

    def layer_sensitivity_heatmap(self, figsize=(12, 6)) -> Figure:
        """
        Heatmap: layers × sparsity levels → performance score.

        Reveals which layers can be aggressively pruned vs which are critical.
        """
        sensitivity = self.tracker._sensitivity_cache

        if not sensitivity:
            fig, ax = plt.subplots(figsize=figsize)
            _apply_style(fig, ax)
            ax.text(0.5, 0.5, "Run tracker.layer_sensitivity() first",
                    ha="center", va="center", color=PALETTE["text"])
            return fig

        fig, ax = plt.subplots(figsize=figsize)
        _apply_style(fig, ax)

        layer_names = list(sensitivity.keys())
        # Get sparsity levels from first layer
        sparsity_levels = [sp for sp, _ in sensitivity[layer_names[0]]]

        # Build matrix: rows=layers, cols=sparsity, values=performance drop %
        baseline = max(sc for _, sc in sensitivity[layer_names[0]])
        matrix = np.zeros((len(layer_names), len(sparsity_levels)))

        for i, layer in enumerate(layer_names):
            for j, (sp, score) in enumerate(sensitivity[layer]):
                drop_pct = (baseline - score) / max(abs(baseline), 1e-10) * 100
                matrix[i, j] = drop_pct

        im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn_r", interpolation="nearest")
        cbar = fig.colorbar(im, ax=ax, label="Performance Drop (%)")
        cbar.ax.yaxis.label.set_color(PALETTE["text"])
        cbar.ax.tick_params(colors=PALETTE["text"])

        # Labels
        short_names = [_short_name(n) for n in layer_names]
        ax.set_yticks(range(len(short_names)))
        ax.set_yticklabels(short_names, fontsize=7)
        ax.set_xticks(range(len(sparsity_levels)))
        ax.set_xticklabels([f"{s:.0%}" for s in sparsity_levels], fontsize=8)
        ax.set_xlabel("Pruning Sparsity")
        ax.set_ylabel("Layer")
        ax.set_title("Layer Sensitivity to Pruning")

        # Annotate cells
        for i in range(len(layer_names)):
            for j in range(len(sparsity_levels)):
                val = matrix[i, j]
                color = "white" if val > matrix.max() * 0.5 else PALETTE["text"]
                ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                       fontsize=6, color=color)

        fig.tight_layout()
        return fig

    def compression_timeline(self, figsize=(12, 5)) -> Figure:
        """
        Bar chart comparing all snapshots: size, params, and performance.
        """
        snaps = self.tracker.snapshots

        if not snaps:
            fig, ax = plt.subplots(figsize=figsize)
            _apply_style(fig, ax)
            ax.text(0.5, 0.5, "No snapshots recorded", ha="center", va="center",
                    color=PALETTE["text"])
            return fig

        fig, axes = plt.subplots(1, 3, figsize=figsize)
        _apply_style(fig, axes)

        names = [s.name[:15] for s in snaps]
        x = range(len(names))

        # 1) Model size
        sizes = [s.model_size_mb for s in snaps]
        axes[0].bar(x, sizes, color=PALETTE["primary"], alpha=0.85)
        axes[0].set_ylabel("Size (MB)")
        axes[0].set_title("Model Size")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(names, rotation=45, ha="right", fontsize=6)

        # 2) Non-zero params
        params = [s.nonzero_params / 1000 for s in snaps]
        axes[1].bar(x, params, color=PALETTE["info"], alpha=0.85)
        axes[1].set_ylabel("Params (K)")
        axes[1].set_title("Active Parameters")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(names, rotation=45, ha="right", fontsize=6)

        # 3) Performance
        scores = [s.eval_metrics.get("score", 0) for s in snaps]
        colors = []
        orig_score = scores[0] if scores else 1
        for s in scores:
            ratio = s / orig_score if orig_score > 0 else 1
            if ratio >= 0.95:
                colors.append(PALETTE["success"])
            elif ratio >= 0.90:
                colors.append(PALETTE["warning"])
            else:
                colors.append(PALETTE["danger"])

        axes[2].bar(x, scores, color=colors, alpha=0.85)
        axes[2].set_ylabel("Score")
        axes[2].set_title("Performance")
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(names, rotation=45, ha="right", fontsize=6)
        # Floor line
        if orig_score > 0:
            axes[2].axhline(y=orig_score * 0.95, color=PALETTE["danger"],
                          linestyle=":", alpha=0.7, label="95% floor")
            axes[2].legend(fontsize=6, facecolor=PALETTE["bg"],
                         edgecolor=PALETTE["grid"], labelcolor=PALETTE["text"])

        fig.suptitle("Compression Timeline", color=PALETTE["text"], fontsize=14)
        fig.tight_layout()
        return fig
