"""
Velocity & acceleration analysis for DL layer parameters.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from flowgrad.snapshot import SnapshotStore


def velocity_per_layer(store: SnapshotStore) -> Dict[str, List[float]]:
    """Return velocity time series for each layer."""
    result = {}
    for name in store.layer_names:
        result[name] = store.get_layer_series(name, "velocity")
    return result


def acceleration_per_layer(store: SnapshotStore) -> Dict[str, List[float]]:
    """Return acceleration time series for each layer."""
    result = {}
    for name in store.layer_names:
        result[name] = store.get_layer_series(name, "acceleration")
    return result


def velocity_heatmap_data(store: SnapshotStore) -> Tuple[np.ndarray, List[str], List[int]]:
    """
    Build 2D array for velocity heatmap: rows=layers, cols=steps.

    Returns:
        (data, layer_names, step_indices)
    """
    layer_names = store.layer_names
    steps = list(range(1, store.num_steps + 1))
    data = np.zeros((len(layer_names), len(steps)))
    for i, name in enumerate(layer_names):
        series = store.get_layer_series(name, "velocity")
        data[i, :len(series)] = series
    return data, layer_names, steps


def detect_stagnation(
    store: SnapshotStore,
    threshold: float = 1e-7,
    min_consecutive: int = 5,
) -> List[Dict]:
    """
    Detect layers where velocity has been below threshold for N consecutive steps.

    Returns:
        List of {name, stagnant_since_step, current_velocity}
    """
    alerts = []
    for name in store.layer_names:
        series = store.get_layer_series(name, "velocity")
        if len(series) < min_consecutive:
            continue
        tail = series[-min_consecutive:]
        if all(v < threshold for v in tail):
            alerts.append({
                "name": name,
                "stagnant_since_step": store.num_steps - min_consecutive + 1,
                "current_velocity": tail[-1],
            })
    return alerts


def detect_explosion(
    store: SnapshotStore,
    threshold: float = 100.0,
) -> List[Dict]:
    """
    Detect layers where gradient norm or velocity exceeds threshold.

    Returns:
        List of {name, step, type, value}
    """
    alerts = []
    for name in store.layer_names:
        grad_norms = store.get_layer_series(name, "grad_norm")
        velocities = store.get_layer_series(name, "velocity")

        for step_idx, gn in enumerate(grad_norms):
            if gn > threshold:
                alerts.append({
                    "name": name,
                    "step": step_idx + 1,
                    "type": "gradient_explosion",
                    "value": gn,
                })

        for step_idx, v in enumerate(velocities):
            if v > threshold:
                alerts.append({
                    "name": name,
                    "step": step_idx + 1,
                    "type": "velocity_explosion",
                    "value": v,
                })

    return alerts
