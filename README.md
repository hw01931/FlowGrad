<p align="center">
  <h1 align="center">🌊 FlowGrad</h1>
  <p align="center">
    <strong>One-line training diagnostics & feature engineering for Deep Learning, Machine Learning, and RecSys.</strong>
  </p>
  <p align="center">
    <a href="#installation">Installation</a> •
    <a href="#quick-start">Quick Start</a> •
    <a href="#features">Features</a> •
    <a href="#examples">Examples</a> •
    <a href="#api-reference">API</a>
  </p>
</p>

---

**FlowGrad** tracks your model's training dynamics in real time, detects hidden feature synergies, and prescribes fixes — all with a single line of code.

```python
tracker = FlowTracker(model)          # That's it. Training diagnostics enabled.
analyzer = FeatureAnalyzer(model, X, y)  # Discover features you didn't know existed.
```

## Why FlowGrad?

| What you do today | What FlowGrad does differently |
|---|---|
| `df.corr()` — linear correlation only | **Non-linear interaction & synergy** detection |
| `model.feature_importances_` — static, post-hoc | **Real-time** importance tracking across rounds |
| SHAP — "why this prediction?" | **"What features should I create?"** — actionable suggestions |
| TensorBoard — manual logging, boilerplate | **Zero-config**, one-line setup |

## Installation

```bash
# From GitHub
pip install git+https://github.com/hw01931/FlowGrad.git

# With all optional dependencies (PyTorch, XGBoost, LightGBM, CatBoost, sklearn)
pip install "flowgrad[all] @ git+https://github.com/hw01931/FlowGrad.git"
```

## Quick Start

### 🔬 Deep Learning (PyTorch)

Track layer-wise weight velocity, gradient health, and dead neurons automatically.

```python
from flowgrad import FlowTracker

tracker = FlowTracker(model)

for epoch in range(100):
    loss = train_one_epoch(model, loader, optimizer)
    tracker.step(loss=loss.item())   # ← just add this line

tracker.report()                     # text diagnostics with prescriptions
tracker.plot.full_report()           # 6-panel visual dashboard
```

<details>
<summary>📊 Available DL Plots</summary>

| Method | What it shows |
|---|---|
| `plot.loss()` | Training loss curve |
| `plot.velocity_heatmap()` | Layer × Step heatmap — hot = fast learning |
| `plot.gradient_flow()` | Per-layer gradient magnitude (detects vanishing/exploding) |
| `plot.weight_distribution()` | Weight norm & std evolution per layer |
| `plot.health_dashboard()` | 0–100 health score per layer |
| `plot.gradient_snr()` | Gradient signal-to-noise ratio over time |
| `plot.full_report()` | All-in-one 6-panel dashboard |

</details>

### 🌲 Gradient Boosting (XGBoost / LightGBM / CatBoost)

Track feature importance drift, evaluation metrics, and overfitting — per round.

```python
from flowgrad import BoostingTracker
import xgboost as xgb

tracker = BoostingTracker()
model = xgb.train(
    params, dtrain,
    num_boost_round=500,
    evals=[(dtrain, "train"), (dval, "valid")],
    callbacks=[tracker.as_xgb_callback()],   # ← just add this
)

tracker.report()
tracker.plot.overfitting_detector()
```

```python
# LightGBM — same interface
lgb.train(params, dtrain, callbacks=[tracker.as_lgb_callback()])

# CatBoost — same interface
model.fit(X, y, callbacks=[tracker.as_catboost_callback()])
```

<details>
<summary>📊 Available Boosting Plots</summary>

| Method | What it shows |
|---|---|
| `plot.eval_metrics()` | Train/valid metric curves |
| `plot.feature_drift()` | Feature importance change over rounds |
| `plot.feature_importance_heatmap()` | Feature × Round heatmap |
| `plot.overfitting_detector()` | Train-valid gap with overfitting zone highlight |
| `plot.full_report()` | All-in-one 4-panel dashboard |

</details>

### 🔧 scikit-learn

Works with GradientBoosting (warm_start), RandomForest (per-tree), and any `partial_fit` model.

```python
from flowgrad import SklearnTracker
from sklearn.ensemble import GradientBoostingClassifier

# Warm-start tracking (GradientBoosting)
tracker = SklearnTracker(feature_names=feature_names)
model = GradientBoostingClassifier(n_estimators=200, warm_start=True)
tracker.track_warm_start(model, X_train, y_train, X_val, y_val, step_size=10)
tracker.report()

# Per-tree analysis (RandomForest)
tracker = SklearnTracker.from_forest(fitted_rf_model, feature_names=feature_names)
tracker.plot.feature_drift()

# Incremental learning (SGDClassifier, etc.)
tracker = SklearnTracker()
tracker.track_partial_fit(model, X_batches, y_batches, classes=[0, 1])
```

### 🧪 Feature Engineering — _The Differentiator_

Go beyond static importance. Discover **feature interactions**, get **concrete combination suggestions**, and detect **redundant features**.

```python
from flowgrad import FeatureAnalyzer

analyzer = FeatureAnalyzer(model, X_train, y_train, feature_names=feature_names)

# 1. Feature Interactions — which pairs have synergy?
interactions = analyzer.interactions(top_k=10)
# → [{"feat_a": "age", "feat_b": "income", "synergy_score": +0.12}, ...]

# 2. Feature Suggestions — what new features should I create?
suggestions = analyzer.suggest_features(top_k=10)
# → [{"expression": "age * income", "lift": +0.08, "target_correlation": 0.72}, ...]

# 3. Redundancy Detection — which features are near-duplicates?
redundant = analyzer.redundant_features(threshold=0.95)
# → [{"feat_a": "height_cm", "feat_b": "height_inch", "recommendation": "Drop height_inch"}]

# 4. Feature Clustering — group related features
clusters = analyzer.feature_clusters()
# → [{"cluster_id": 0, "features": ["age", "income"], "cohesion": 0.85}]

# Full text report with all of the above
analyzer.report()
```

<details>
<summary>📊 Available Feature Engineering Plots</summary>

| Method | What it shows |
|---|---|
| `plot.interaction_heatmap()` | Feature pair synergy matrix |
| `plot.suggestion_chart()` | Top combinations ranked by lift |
| `plot.redundancy_graph()` | Network graph of redundant feature pairs |
| `plot.cluster_map()` | Grouped feature visualization |

</details>

## Features

### 🩺 Automated Diagnostics

FlowGrad doesn't just visualize — it **diagnoses problems and prescribes fixes**.

```
⚠️  Alerts & Prescriptions
─────────────────────────────────────────────
  🧊 STAGNATION: 'layer3.conv.weight'
     Velocity ≈ 1.2e-09 since step 45
     💊 Increase learning rate or remove weight decay for this layer.

  💥 GRADIENT_EXPLOSION: 'layer1.fc.weight'
     Value: 512.3 at step 12
     💊 Add gradient clipping (max_norm=1.0) or reduce lr by 50%.

  💀 DEAD NEURONS: 'layer2.relu.weight'
     62.3% of parameters are near-zero
     💊 Consider LeakyReLU or PReLU. Check initialization.
```

### What's tracked

| Domain | Metrics |
|---|---|
| **DL Layers** | Weight norm/mean/std, gradient norm/SNR, velocity (ΔW), acceleration (ΔΔW), dead neuron ratio |
| **Boosting Rounds** | Per-round feature importance, train/valid eval metrics, overfitting gap |
| **Feature Engineering** | Pairwise synergy, combination lift, redundancy correlation, cluster cohesion |

## Examples

📓 **[Colab Demo Notebook](examples/demo_colab.ipynb)** — Run all features on real data (sklearn built-in datasets, zero setup required).

## API Reference

| Class | Purpose | Input |
|---|---|---|
| `FlowTracker(model)` | DL training dynamics | PyTorch `nn.Module` |
| `BoostingTracker()` | Boosting round tracking | XGBoost / LightGBM / CatBoost callback |
| `SklearnTracker()` | sklearn model tracking | Any sklearn estimator |
| `FeatureAnalyzer(model, X, y)` | Feature engineering analysis | Any fitted model + data |

Every tracker exposes:
- `.report()` → text diagnostics
- `.plot.*` → matplotlib figures
- `.summary` → dict for programmatic access
- `.history` → raw collected data

## Roadmap

- [x] v0.1 — DL Tracker + Boosting Tracker
- [x] v0.2 — scikit-learn support + Feature Engineering
- [ ] v0.3 — RecSys module (embedding drift, coverage, cold start)
- [ ] v0.4 — Plotly interactive dashboards
- [ ] v0.5 — PyPI release

## License

[MIT](LICENSE)
