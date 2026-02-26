"""Tests for BoostingTracker."""
import pytest

from flowgrad.analyzers.boosting import BoostingTracker
from flowgrad.snapshot import BoostingRoundRecord


class TestBoostingTrackerManual:
    """Test BoostingTracker with manual step() calls (framework-agnostic)."""

    def test_manual_step(self):
        tracker = BoostingTracker()
        tracker.step(
            round_num=1,
            eval_metrics={"train": {"rmse": 0.5}, "valid": {"rmse": 0.6}},
            feature_importance={"feat_a": 100, "feat_b": 50, "feat_c": 10},
        )
        assert tracker.store.num_rounds == 1

    def test_multiple_rounds(self):
        tracker = BoostingTracker()
        for i in range(10):
            tracker.step(
                round_num=i + 1,
                eval_metrics={"train": {"rmse": 1.0 / (i + 1)}},
                feature_importance={"feat_a": 100 + i * 10, "feat_b": 50 - i * 2},
            )
        assert tracker.store.num_rounds == 10

    def test_feature_importance_series(self):
        tracker = BoostingTracker()
        for i in range(5):
            tracker.step(
                round_num=i + 1,
                feature_importance={"feat_a": float(i * 10), "feat_b": float(50 - i)},
            )
        series_a = tracker.store.get_feature_importance_series("feat_a")
        assert series_a == [0.0, 10.0, 20.0, 30.0, 40.0]

    def test_eval_metric_series(self):
        tracker = BoostingTracker()
        for i in range(5):
            tracker.step(
                round_num=i + 1,
                eval_metrics={"train": {"rmse": 1.0 / (i + 1)}},
            )
        series = tracker.store.get_eval_metric_series("train", "rmse")
        assert len(series) == 5
        assert series[0] == pytest.approx(1.0)
        assert series[-1] == pytest.approx(0.2)

    def test_summary_keys(self):
        tracker = BoostingTracker()
        for i in range(5):
            tracker.step(
                round_num=i + 1,
                eval_metrics={"train": {"rmse": 1.0 / (i + 1)}},
                feature_importance={"a": float(i), "b": float(10 - i)},
            )
        s = tracker.summary
        assert "total_rounds" in s
        assert s["total_rounds"] == 5
        assert "features_tracked" in s

    def test_repr(self):
        tracker = BoostingTracker()
        r = repr(tracker)
        assert "BoostingTracker" in r


class TestBoostingTrackerCallbackCreation:
    """Test that callback objects can be created (without actual frameworks)."""

    def test_xgb_callback_creation(self):
        """Only test creation, not usage (requires xgboost)."""
        tracker = BoostingTracker()
        try:
            cb = tracker.as_xgb_callback()
            assert cb is not None
        except ImportError:
            pytest.skip("XGBoost not installed")

    def test_lgb_callback_creation(self):
        """Only test creation."""
        tracker = BoostingTracker()
        try:
            cb = tracker.as_lgb_callback()
            assert callable(cb)
        except ImportError:
            pytest.skip("LightGBM not installed")

    def test_catboost_callback_creation(self):
        """Only test creation."""
        tracker = BoostingTracker()
        try:
            cb = tracker.as_catboost_callback()
            assert hasattr(cb, "after_iteration")
        except ImportError:
            pytest.skip("CatBoost not installed")


class TestBoostingReport:
    def test_report_runs(self):
        tracker = BoostingTracker()
        for i in range(10):
            tracker.step(
                round_num=i + 1,
                eval_metrics={"train": {"rmse": 1.0 / (i + 1)}, "valid": {"rmse": 1.2 / (i + 1)}},
                feature_importance={"feat_a": float(100 + i * 10), "feat_b": float(50 - i * 2)},
            )
        report = tracker.report()
        assert "FlowGrad" in report
        assert "feat_a" in report
