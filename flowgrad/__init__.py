"""
FlowGrad — ML Training Dynamics Tracker

One-line layer-by-layer visualization for PyTorch & boosting models.

Usage (PyTorch):
    from flowgrad import FlowTracker
    tracker = FlowTracker(model)
    for epoch in range(100):
        loss = train(model)
        tracker.step(loss=loss)
    tracker.report()

Usage (XGBoost / LightGBM / CatBoost):
    from flowgrad import BoostingTracker
    tracker = BoostingTracker()
    model = xgb.train(params, dtrain, callbacks=[tracker.as_xgb_callback()])
    tracker.report()
"""

__version__ = "0.3.0"

from flowgrad.tracker import FlowTracker
from flowgrad.analyzers.boosting import BoostingTracker
from flowgrad.analyzers.sklearn_tracker import SklearnTracker
from flowgrad.analyzers.features import FeatureAnalyzer
from flowgrad.analyzers.compression import CompressionTracker

__all__ = [
    "FlowTracker",
    "BoostingTracker",
    "SklearnTracker",
    "FeatureAnalyzer",
    "CompressionTracker",
    "__version__",
]
