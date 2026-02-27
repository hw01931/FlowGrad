"""Tests for velocity and health analyzers."""
import pytest
import math

from gradtracer.snapshot import LayerSnapshot, SnapshotStore, StepRecord


def _make_store_with_steps(n_steps=10, n_layers=3):
    """Create a test store with synthetic data."""
    store = SnapshotStore()
    for step in range(1, n_steps + 1):
        record = StepRecord(step=step, loss=1.0 / step)
        for li in range(n_layers):
            name = f"layer_{li}.weight"
            snap = LayerSnapshot(
                name=name,
                step=step,
                weight_norm=1.0 + step * 0.1 * (li + 1),
                weight_mean=0.01 * step,
                weight_std=0.5 + step * 0.01,
                grad_norm=0.1 / (step + 1),
                grad_mean=0.001 * (li + 1),
                grad_std=0.01,
                velocity=0.01 / (step + 1),
                acceleration=0.001 / (step + 1),
                dead_ratio=0.1 * li,
            )
            record.layers[name] = snap
        store.add_step(record)
    return store


def _make_store_stagnant():
    """Create a store where layer_0 is stagnant."""
    store = SnapshotStore()
    for step in range(1, 15):
        record = StepRecord(step=step, loss=1.0)
        for li in range(2):
            name = f"layer_{li}.weight"
            snap = LayerSnapshot(
                name=name, step=step,
                velocity=1e-10 if li == 0 else 0.05,
                grad_norm=0.1, grad_mean=0.01, grad_std=0.01,
                weight_norm=1.0, weight_mean=0.0, weight_std=0.5,
            )
            record.layers[name] = snap
        store.add_step(record)
    return store


def _make_store_exploding():
    """Create a store where layer_1 has gradient explosion."""
    store = SnapshotStore()
    for step in range(1, 6):
        record = StepRecord(step=step, loss=1.0)
        for li in range(2):
            name = f"layer_{li}.weight"
            snap = LayerSnapshot(
                name=name, step=step,
                velocity=0.1,
                grad_norm=500.0 if li == 1 else 0.5,
                grad_mean=0.01, grad_std=0.01,
                weight_norm=1.0, weight_mean=0.0, weight_std=0.5,
            )
            record.layers[name] = snap
        store.add_step(record)
    return store


class TestVelocityAnalyzer:
    def test_velocity_per_layer(self):
        from gradtracer.analyzers.velocity import velocity_per_layer
        store = _make_store_with_steps()
        result = velocity_per_layer(store)
        assert len(result) == 3
        assert all(len(v) == 10 for v in result.values())

    def test_velocity_heatmap_data(self):
        from gradtracer.analyzers.velocity import velocity_heatmap_data
        store = _make_store_with_steps()
        data, names, steps = velocity_heatmap_data(store)
        assert data.shape == (3, 10)
        assert len(names) == 3
        assert len(steps) == 10

    def test_detect_stagnation(self):
        from gradtracer.analyzers.velocity import detect_stagnation
        store = _make_store_stagnant()
        alerts = detect_stagnation(store, threshold=1e-7, min_consecutive=5)
        assert len(alerts) >= 1
        assert alerts[0]["name"] == "layer_0.weight"

    def test_no_false_stagnation(self):
        from gradtracer.analyzers.velocity import detect_stagnation
        store = _make_store_with_steps()
        alerts = detect_stagnation(store, threshold=1e-15)
        assert len(alerts) == 0

    def test_detect_explosion(self):
        from gradtracer.analyzers.velocity import detect_explosion
        store = _make_store_exploding()
        alerts = detect_explosion(store, threshold=100.0)
        assert len(alerts) >= 1
        exploding_names = {a["name"] for a in alerts}
        assert "layer_1.weight" in exploding_names


class TestHealthAnalyzer:
    def test_gradient_snr(self):
        from gradtracer.analyzers.health import gradient_snr_per_layer
        store = _make_store_with_steps()
        snr = gradient_snr_per_layer(store)
        assert len(snr) == 3
        # SNR = mean^2 / std^2, should be > 0
        for name, series in snr.items():
            assert all(s >= 0 for s in series)

    def test_dead_neuron_ratio(self):
        from gradtracer.analyzers.health import dead_neuron_ratio_per_layer
        store = _make_store_with_steps()
        result = dead_neuron_ratio_per_layer(store)
        assert len(result) == 3
        # layer_0 should have 0 dead ratio, layer_2 should have 0.2
        assert result["layer_0.weight"][0] == pytest.approx(0.0)
        assert result["layer_2.weight"][0] == pytest.approx(0.2)

    def test_health_score_range(self):
        from gradtracer.analyzers.health import layer_health_score
        store = _make_store_with_steps()
        scores = layer_health_score(store)
        assert len(scores) == 3
        for name, score in scores.items():
            assert 0 <= score <= 100
