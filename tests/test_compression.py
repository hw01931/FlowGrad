"""Tests for CompressionTracker."""
import pytest
import numpy as np


def _make_model_and_data():
    """Create a simple PyTorch model and data for testing."""
    import torch
    import torch.nn as nn

    model = nn.Sequential(
        nn.Linear(20, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
    )

    X = torch.randn(200, 20)
    y = (X[:, 0] + X[:, 1] > 0).float()
    return model, X, y


def _train_model(model, X, y, epochs=20):
    """Train model for a few epochs."""
    import torch
    import torch.nn as nn

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for _ in range(epochs):
        optimizer.zero_grad()
        out = model(X).squeeze()
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
    return model


def _make_eval_fn(X, y):
    """Create an evaluation function."""
    import torch

    def eval_fn(model):
        model.eval()
        with torch.no_grad():
            preds = model(X).squeeze()
            pred_labels = (preds > 0).float()
            acc = (pred_labels == y).float().mean().item()
        model.train()
        return acc

    return eval_fn


@pytest.fixture
def setup():
    """Set up model, data, and eval_fn."""
    model, X, y = _make_model_and_data()
    model = _train_model(model, X, y)
    eval_fn = _make_eval_fn(X, y)
    return model, X, y, eval_fn


class TestCompressionSnapshot:
    def test_snapshot_records_stats(self, setup):
        model, X, y, eval_fn = setup
        from gradtracer import CompressionTracker

        tracker = CompressionTracker(model, eval_fn=eval_fn)
        snap = tracker.snapshot("original")

        assert snap.name == "original"
        assert snap.total_params > 0
        assert snap.nonzero_params > 0
        assert snap.model_size_mb > 0
        assert snap.sparsity == 0.0  # no pruning yet
        assert "score" in snap.eval_metrics
        assert len(snap.layer_stats) > 0

    def test_multiple_snapshots(self, setup):
        model, X, y, eval_fn = setup
        from gradtracer import CompressionTracker

        tracker = CompressionTracker(model, eval_fn=eval_fn)
        tracker.snapshot("original")

        # Apply manual pruning
        CompressionTracker._apply_pruning(model, 0.3)
        tracker.snapshot("pruned_30%", sparsity=0.3)

        assert len(tracker.snapshots) == 2
        assert tracker.snapshots[1].sparsity > 0


class TestAutoCompress:
    def test_binary_search_pruning(self, setup):
        model, X, y, eval_fn = setup
        from gradtracer import CompressionTracker

        tracker = CompressionTracker(model, eval_fn=eval_fn)
        result = tracker.auto_compress(
            method="pruning",
            performance_floor=0.90,
            search_range=(0.1, 0.7),
            search_strategy="binary",
            precision=0.1,
        )

        assert result.performance_retained > 0
        assert result.size_reduction >= 0
        assert len(result.all_snapshots) >= 2
        assert result.recommendation != ""
        assert "pruning" in result.recommendation.lower() or "config" in result.recommendation.lower()

    def test_grid_search_pruning(self, setup):
        model, X, y, eval_fn = setup
        from gradtracer import CompressionTracker

        tracker = CompressionTracker(model, eval_fn=eval_fn)
        result = tracker.auto_compress(
            method="pruning",
            performance_floor=0.85,
            search_range=(0.1, 0.5),
            search_strategy="grid",
        )

        # Grid with 9 steps + 1 original = 10
        assert len(result.all_snapshots) >= 5


class TestLayerSensitivity:
    def test_sensitivity_profiling(self, setup):
        model, X, y, eval_fn = setup
        from gradtracer import CompressionTracker

        tracker = CompressionTracker(model, eval_fn=eval_fn)
        results = tracker.layer_sensitivity(
            sparsity_levels=[0.1, 0.3, 0.5],
        )

        assert len(results) > 0
        # Each layer should have 3 entries (one per sparsity)
        for layer_name, entries in results.items():
            assert len(entries) == 3

    def test_nonuniform_recommendation(self, setup):
        model, X, y, eval_fn = setup
        from gradtracer import CompressionTracker

        tracker = CompressionTracker(model, eval_fn=eval_fn)
        tracker.layer_sensitivity(sparsity_levels=[0.1, 0.3, 0.5])
        rec = tracker.recommend_nonuniform(performance_floor=0.90)

        assert len(rec) > 0
        for layer_name, sparsity in rec.items():
            assert 0 <= sparsity <= 1


class TestCompressionReport:
    def test_report_basic(self, setup):
        model, X, y, eval_fn = setup
        from gradtracer import CompressionTracker

        tracker = CompressionTracker(model, eval_fn=eval_fn)
        tracker.snapshot("original")
        CompressionTracker._apply_pruning(model, 0.3)
        tracker.snapshot("pruned_30%")

        report = tracker.report()
        assert "GradTracer" in report
        assert "original" in report
        assert "pruned_30%" in report

    def test_report_with_sensitivity(self, setup):
        model, X, y, eval_fn = setup
        from gradtracer import CompressionTracker

        tracker = CompressionTracker(model, eval_fn=eval_fn)
        tracker.snapshot("original")
        tracker.layer_sensitivity(sparsity_levels=[0.1, 0.3])

        report = tracker.report()
        assert "Sensitivity" in report


class TestCompressionPlots:
    def test_tradeoff_curve(self, setup):
        model, X, y, eval_fn = setup
        from gradtracer import CompressionTracker
        import matplotlib
        matplotlib.use("Agg")

        tracker = CompressionTracker(model, eval_fn=eval_fn)
        tracker.snapshot("original")
        CompressionTracker._apply_pruning(model, 0.3)
        tracker.snapshot("pruned_30%")

        fig = tracker.plot.tradeoff_curve()
        assert fig is not None

    def test_compression_timeline(self, setup):
        model, X, y, eval_fn = setup
        from gradtracer import CompressionTracker
        import matplotlib
        matplotlib.use("Agg")

        tracker = CompressionTracker(model, eval_fn=eval_fn)
        tracker.snapshot("original")

        fig = tracker.plot.compression_timeline()
        assert fig is not None

    def test_sensitivity_heatmap(self, setup):
        model, X, y, eval_fn = setup
        from gradtracer import CompressionTracker
        import matplotlib
        matplotlib.use("Agg")

        tracker = CompressionTracker(model, eval_fn=eval_fn)
        tracker.layer_sensitivity(sparsity_levels=[0.1, 0.3, 0.5])

        fig = tracker.plot.layer_sensitivity_heatmap()
        assert fig is not None


class TestVIFCollinearity:
    def test_suggest_features_with_vif(self):
        from gradtracer import FeatureAnalyzer
        from sklearn.ensemble import RandomForestRegressor

        rng = np.random.RandomState(42)
        X = rng.randn(200, 5)
        y = X[:, 0] * X[:, 1] + rng.randn(200) * 0.3
        names = [f"f{i}" for i in range(5)]

        model = RandomForestRegressor(n_estimators=20, random_state=42).fit(X, y)
        analyzer = FeatureAnalyzer(model, X, y, feature_names=names)

        suggestions = analyzer.suggest_features(
            top_k=5,
            collinearity_check=True,
            vif_threshold=10.0,
        )

        assert len(suggestions) > 0
        # All positive-lift suggestions should have vif_score
        for s in suggestions:
            if s["lift"] > 0:
                assert "vif_score" in s
                assert "collinearity_warning" in s

    def test_no_collinearity_check(self):
        from gradtracer import FeatureAnalyzer
        from sklearn.ensemble import RandomForestRegressor

        rng = np.random.RandomState(42)
        X = rng.randn(100, 3)
        y = X[:, 0] + rng.randn(100) * 0.5

        model = RandomForestRegressor(n_estimators=10, random_state=42).fit(X, y)
        analyzer = FeatureAnalyzer(model, X, y)

        suggestions = analyzer.suggest_features(top_k=3, collinearity_check=False)
        assert len(suggestions) > 0
        # Should not have vif fields
        assert "vif_score" not in suggestions[0]
