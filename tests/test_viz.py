"""Smoke tests for visualization module."""
import pytest
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for testing

from flowgrad.snapshot import LayerSnapshot, SnapshotStore, StepRecord
from flowgrad.snapshot import BoostingStore, BoostingRoundRecord


def _make_dl_store(n_steps=10, n_layers=3):
    store = SnapshotStore()
    for step in range(1, n_steps + 1):
        record = StepRecord(step=step, loss=1.0 / step)
        for li in range(n_layers):
            name = f"layer_{li}.weight"
            snap = LayerSnapshot(
                name=name, step=step,
                weight_norm=1.0 + step * 0.1,
                weight_mean=0.01 * step,
                weight_std=0.5,
                grad_norm=0.1 / (step + 1),
                grad_mean=0.001,
                grad_std=0.01,
                velocity=0.01 / (step + 1),
                acceleration=0.001 / (step + 1),
                dead_ratio=0.05 * li,
            )
            record.layers[name] = snap
        store.add_step(record)
    return store


def _make_boosting_store(n_rounds=20, n_features=5):
    store = BoostingStore()
    for r in range(1, n_rounds + 1):
        record = BoostingRoundRecord(
            round=r,
            eval_metrics={
                "train": {"rmse": 1.0 / r},
                "valid": {"rmse": 1.2 / r + 0.01 * r},  # diverges later
            },
            feature_importance={
                f"feature_{i}": float(10 * (i + 1) + r * (i - 2))
                for i in range(n_features)
            },
        )
        store.add_round(record)
    return store


class TestDLPlots:
    def test_loss_plot(self):
        from flowgrad.viz.plots import DLPlotAPI
        store = _make_dl_store()
        api = DLPlotAPI(store)
        fig = api.loss()
        assert fig is not None

    def test_velocity_heatmap(self):
        from flowgrad.viz.plots import DLPlotAPI
        store = _make_dl_store()
        api = DLPlotAPI(store)
        fig = api.velocity_heatmap()
        assert fig is not None

    def test_gradient_flow(self):
        from flowgrad.viz.plots import DLPlotAPI
        store = _make_dl_store()
        api = DLPlotAPI(store)
        fig = api.gradient_flow()
        assert fig is not None

    def test_weight_distribution(self):
        from flowgrad.viz.plots import DLPlotAPI
        store = _make_dl_store()
        api = DLPlotAPI(store)
        fig = api.weight_distribution()
        assert fig is not None

    def test_health_dashboard(self):
        from flowgrad.viz.plots import DLPlotAPI
        store = _make_dl_store()
        api = DLPlotAPI(store)
        fig = api.health_dashboard()
        assert fig is not None

    def test_gradient_snr(self):
        from flowgrad.viz.plots import DLPlotAPI
        store = _make_dl_store()
        api = DLPlotAPI(store)
        fig = api.gradient_snr()
        assert fig is not None

    def test_full_report(self):
        from flowgrad.viz.plots import DLPlotAPI
        store = _make_dl_store()
        api = DLPlotAPI(store)
        fig = api.full_report()
        assert fig is not None


class TestBoostingPlots:
    def test_eval_metrics(self):
        from flowgrad.viz.plots import BoostingPlotAPI
        store = _make_boosting_store()
        api = BoostingPlotAPI(store)
        fig = api.eval_metrics()
        assert fig is not None

    def test_feature_drift(self):
        from flowgrad.viz.plots import BoostingPlotAPI
        store = _make_boosting_store()
        api = BoostingPlotAPI(store)
        fig = api.feature_drift()
        assert fig is not None

    def test_feature_importance_heatmap(self):
        from flowgrad.viz.plots import BoostingPlotAPI
        store = _make_boosting_store()
        api = BoostingPlotAPI(store)
        fig = api.feature_importance_heatmap()
        assert fig is not None

    def test_overfitting_detector(self):
        from flowgrad.viz.plots import BoostingPlotAPI
        store = _make_boosting_store()
        api = BoostingPlotAPI(store)
        fig = api.overfitting_detector()
        assert fig is not None

    def test_full_report(self):
        from flowgrad.viz.plots import BoostingPlotAPI
        store = _make_boosting_store()
        api = BoostingPlotAPI(store)
        fig = api.full_report()
        assert fig is not None
