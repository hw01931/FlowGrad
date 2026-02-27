"""
Layer health diagnostics — gradient SNR, dead neurons, health scoring.
"""
from __future__ import annotations

import math
from typing import Dict, List

import numpy as np

from gradtracer.snapshot import SnapshotStore


def gradient_snr_per_layer(store: SnapshotStore) -> Dict[str, List[float]]:
    """
    Compute Gradient Signal-to-Noise Ratio per layer over time.

    SNR = mean(grad)^2 / var(grad).
    Higher SNR → more consistent gradient direction → more useful learning signal.
    """
    result = {}
    for name in store.layer_names:
        means = store.get_layer_series(name, "grad_mean")
        stds = store.get_layer_series(name, "grad_std")
        snr = []
        for m, s in zip(means, stds):
            if s > 1e-12:
                snr.append((m ** 2) / (s ** 2))
            else:
                snr.append(float("inf") if abs(m) > 1e-12 else 0.0)
        result[name] = snr
    return result


def dead_neuron_ratio_per_layer(store: SnapshotStore) -> Dict[str, List[float]]:
    """Get dead neuron ratio for each layer over time."""
    result = {}
    for name in store.layer_names:
        result[name] = store.get_layer_series(name, "dead_ratio")
    return result


def layer_health_score(store: SnapshotStore) -> Dict[str, float]:
    """
    Compute a 0-100 health score for each layer based on current state.

    Scoring heuristic (latest step):
        - Gradient norm in healthy range [1e-5, 10]: +40 pts
        - Velocity > 0 (still learning): +20 pts
        - Dead neuron ratio < 0.3: +20 pts
        - Gradient SNR > 0.01: +20 pts

    Returns:
        {layer_name: score}
    """
    snr_data = gradient_snr_per_layer(store)
    scores = {}

    for name in store.layer_names:
        score = 0.0
        history = store.get_layer_history(name)
        if not history:
            scores[name] = 0.0
            continue

        latest = history[-1]

        # Gradient norm health (40 pts)
        gn = latest.grad_norm
        if 1e-5 <= gn <= 10.0:
            score += 40.0
        elif 1e-7 <= gn <= 100.0:
            score += 20.0
        elif gn > 100.0:
            score += 0.0  # explosion
        else:
            score += 5.0  # vanishing

        # Velocity (20 pts): still learning?
        if latest.velocity > 1e-8:
            score += 20.0
        elif latest.velocity > 1e-10:
            score += 10.0

        # Dead neuron ratio (20 pts)
        if latest.dead_ratio < 0.1:
            score += 20.0
        elif latest.dead_ratio < 0.3:
            score += 12.0
        elif latest.dead_ratio < 0.5:
            score += 5.0

        # Gradient SNR (20 pts)
        snr_series = snr_data.get(name, [])
        if snr_series:
            latest_snr = snr_series[-1]
            if not math.isinf(latest_snr):
                if latest_snr > 0.1:
                    score += 20.0
                elif latest_snr > 0.01:
                    score += 12.0
                elif latest_snr > 0.001:
                    score += 5.0

        scores[name] = min(score, 100.0)

    return scores
