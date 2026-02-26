"""Tests for SklearnTracker and FeatureAnalyzer."""
import pytest
import numpy as np


def _make_classification_data(n_samples=200, n_features=5, seed=42):
    """Create a simple classification dataset."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_features)
    # Feature 0 and 1 are actually useful, others are noise
    y = (X[:, 0] * 2 + X[:, 1] * 1.5 + rng.randn(n_samples) * 0.5 > 0).astype(int)
    feature_names = [f"feat_{i}" for i in range(n_features)]
    return X, y, feature_names


def _make_regression_data(n_samples=200, n_features=5, seed=42):
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_features)
    # Target depends on product of feat_0 * feat_1 (interaction!)
    y = X[:, 0] * X[:, 1] + 0.5 * X[:, 2] + rng.randn(n_samples) * 0.3
    feature_names = [f"feat_{i}" for i in range(n_features)]
    return X, y, feature_names


def _make_redundant_data(n_samples=200, seed=42):
    """Create data with redundant features."""
    rng = np.random.RandomState(seed)
    X = np.zeros((n_samples, 5))
    X[:, 0] = rng.randn(n_samples)
    X[:, 1] = X[:, 0] * 1.01 + rng.randn(n_samples) * 0.01  # near-duplicate of 0
    X[:, 2] = rng.randn(n_samples)
    X[:, 3] = rng.randn(n_samples)
    X[:, 4] = X[:, 3] * -0.99 + rng.randn(n_samples) * 0.01  # near-duplicate of 3 (negated)
    y = (X[:, 0] + X[:, 2] > 0).astype(int)
    return X, y, ["feat_a", "feat_a_dup", "feat_b", "feat_c", "feat_c_neg"]


# ======================================================================
#  SklearnTracker Tests
# ======================================================================

class TestSklearnTrackerWarmStart:
    def test_gradient_boosting_classifier(self):
        from sklearn.ensemble import GradientBoostingClassifier
        from flowgrad import SklearnTracker

        X, y, names = _make_classification_data()
        model = GradientBoostingClassifier(
            n_estimators=50, max_depth=3, warm_start=True, random_state=42
        )
        tracker = SklearnTracker(feature_names=names)
        tracker.track_warm_start(model, X, y, step_size=10)

        assert tracker.store.num_rounds == 5  # 10, 20, 30, 40, 50
        assert len(tracker.store.get_all_feature_names()) == 5

    def test_gradient_boosting_regressor(self):
        from sklearn.ensemble import GradientBoostingRegressor
        from flowgrad import SklearnTracker

        X, y, names = _make_regression_data()
        model = GradientBoostingRegressor(
            n_estimators=30, warm_start=True, random_state=42
        )
        tracker = SklearnTracker(feature_names=names)
        tracker.track_warm_start(model, X, y, step_size=10)

        assert tracker.store.num_rounds == 3

    def test_with_validation(self):
        from sklearn.ensemble import GradientBoostingClassifier
        from flowgrad import SklearnTracker

        X, y, names = _make_classification_data(n_samples=300)
        X_train, X_val = X[:200], X[200:]
        y_train, y_val = y[:200], y[200:]

        model = GradientBoostingClassifier(
            n_estimators=20, warm_start=True, random_state=42
        )
        tracker = SklearnTracker(feature_names=names)
        tracker.track_warm_start(model, X_train, y_train, X_val, y_val, step_size=10)

        # Should have validation metrics
        metrics = tracker.store.rounds[0].eval_metrics
        assert "valid" in metrics


class TestSklearnTrackerForest:
    def test_random_forest(self):
        from sklearn.ensemble import RandomForestClassifier
        from flowgrad import SklearnTracker

        X, y, names = _make_classification_data()
        model = RandomForestClassifier(n_estimators=20, random_state=42)
        model.fit(X, y)

        tracker = SklearnTracker.from_forest(model, feature_names=names)
        assert tracker.store.num_rounds == 20  # one per tree


class TestSklearnTrackerPartialFit:
    def test_sgd_classifier(self):
        from sklearn.linear_model import SGDClassifier
        from flowgrad import SklearnTracker

        X, y, names = _make_classification_data(n_samples=500)
        # Split into batches
        batch_size = 100
        X_batches = [X[i:i+batch_size] for i in range(0, 500, batch_size)]
        y_batches = [y[i:i+batch_size] for i in range(0, 500, batch_size)]

        model = SGDClassifier(random_state=42)
        tracker = SklearnTracker(feature_names=names)
        tracker.track_partial_fit(model, X_batches, y_batches, classes=[0, 1])

        assert tracker.store.num_rounds == 5


class TestSklearnTrackerReport:
    def test_report_runs(self):
        from sklearn.ensemble import GradientBoostingClassifier
        from flowgrad import SklearnTracker

        X, y, names = _make_classification_data()
        model = GradientBoostingClassifier(
            n_estimators=30, warm_start=True, random_state=42
        )
        tracker = SklearnTracker(feature_names=names)
        tracker.track_warm_start(model, X, y, step_size=10)

        report = tracker.report()
        assert "FlowGrad" in report


# ======================================================================
#  FeatureAnalyzer Tests
# ======================================================================

class TestFeatureInteractions:
    def test_correlation_method(self):
        from flowgrad.analyzers.features import FeatureAnalyzer
        from sklearn.ensemble import RandomForestRegressor

        X, y, names = _make_regression_data()
        model = RandomForestRegressor(n_estimators=20, random_state=42).fit(X, y)

        analyzer = FeatureAnalyzer(model, X, y, feature_names=names)
        interactions = analyzer.interactions(top_k=5, method="correlation")

        assert len(interactions) > 0
        assert "feat_a" in interactions[0]
        assert "synergy_score" in interactions[0]

    def test_permutation_method(self):
        from flowgrad.analyzers.features import FeatureAnalyzer
        from sklearn.ensemble import RandomForestClassifier

        X, y, names = _make_classification_data()
        model = RandomForestClassifier(n_estimators=20, random_state=42).fit(X, y)

        analyzer = FeatureAnalyzer(model, X, y, feature_names=names)
        interactions = analyzer.interactions(top_k=3, method="permutation")

        assert len(interactions) > 0
        assert "interaction_strength" in interactions[0]


class TestFeatureSuggestions:
    def test_suggest_features(self):
        from flowgrad.analyzers.features import FeatureAnalyzer
        from sklearn.ensemble import RandomForestRegressor

        X, y, names = _make_regression_data()
        model = RandomForestRegressor(n_estimators=20, random_state=42).fit(X, y)

        analyzer = FeatureAnalyzer(model, X, y, feature_names=names)
        suggestions = analyzer.suggest_features(top_k=5)

        assert len(suggestions) > 0
        assert "expression" in suggestions[0]
        assert "lift" in suggestions[0]

    def test_interaction_feature_suggested_for_product_data(self):
        """When y = x0 * x1, the product feature should be highly ranked."""
        from flowgrad.analyzers.features import FeatureAnalyzer
        from sklearn.ensemble import RandomForestRegressor

        X, y, names = _make_regression_data()
        model = RandomForestRegressor(n_estimators=30, random_state=42).fit(X, y)

        analyzer = FeatureAnalyzer(model, X, y, feature_names=names)
        suggestions = analyzer.suggest_features(top_k=20)

        # The product feat_0 * feat_1 should appear with high correlation
        product_suggestions = [
            s for s in suggestions
            if "feat_0" in s["expression"] and "feat_1" in s["expression"]
            and "*" in s["expression"]
        ]
        assert len(product_suggestions) > 0
        assert product_suggestions[0]["target_correlation"] > 0.5


class TestRedundancy:
    def test_detects_redundant_features(self):
        from flowgrad.analyzers.features import FeatureAnalyzer
        from sklearn.ensemble import RandomForestClassifier

        X, y, names = _make_redundant_data()
        model = RandomForestClassifier(n_estimators=20, random_state=42).fit(X, y)

        analyzer = FeatureAnalyzer(model, X, y, feature_names=names)
        redundant = analyzer.redundant_features(threshold=0.95)

        assert len(redundant) >= 2
        pairs = {(r["feat_a"], r["feat_b"]) for r in redundant}
        # Should detect feat_a / feat_a_dup pair
        assert ("feat_a", "feat_a_dup") in pairs or ("feat_a_dup", "feat_a") in pairs


class TestFeatureClusters:
    def test_clusters(self):
        from flowgrad.analyzers.features import FeatureAnalyzer
        from sklearn.ensemble import RandomForestClassifier

        X, y, names = _make_classification_data(n_features=8)
        model = RandomForestClassifier(n_estimators=20, random_state=42).fit(X, y)

        analyzer = FeatureAnalyzer(model, X, y)
        clusters = analyzer.feature_clusters()

        assert len(clusters) >= 2
        total_features = sum(c["size"] for c in clusters)
        assert total_features == 8


class TestFeatureReport:
    def test_full_report(self):
        from flowgrad.analyzers.features import FeatureAnalyzer
        from sklearn.ensemble import RandomForestRegressor

        X, y, names = _make_regression_data()
        model = RandomForestRegressor(n_estimators=20, random_state=42).fit(X, y)

        analyzer = FeatureAnalyzer(model, X, y, feature_names=names)
        report = analyzer.report()

        assert "Feature Engineering Report" in report
        assert "Interactions" in report
        assert "Suggested" in report
