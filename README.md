<p align="center">
  <h1 align="center">🌊 FlowGrad</h1>
  <p align="center">
    <strong>One-line training diagnostics, feature engineering & model compression for Deep Learning, Machine Learning, and RecSys.</strong>
  </p>
  <p align="center">
    <a href="#installation">Installation</a> •
    <a href="#quick-start">Quick Start</a> •
    <a href="#features">Features</a> •
    <a href="#api-reference">API</a>
  </p>
</p>

---

**FlowGrad** tracks your model's training dynamics in real time, detects hidden feature synergies, auto-compresses models, and prescribes fixes — all with a single line of code.

```python
tracker = FlowTracker(model)                     # DL training diagnostics
analyzer = FeatureAnalyzer(model, X, y)           # Feature engineering
comp = CompressionTracker(model, eval_fn)         # Auto compression search
xml = tracker.export_for_agent()                  # AI-native output for Cursor/Copilot
```

## Why FlowGrad?

| What you do today | What FlowGrad does differently |
|---|---|
| `df.corr()` — linear only | **Non-linear interaction & synergy** detection |
| `model.feature_importances_` — static | **Real-time** importance tracking across rounds |
| SHAP — "why this prediction?" | **"What features should I create?"** + VIF filter |
| TensorBoard — manual logging | **Zero-config**, one-line setup |
| NNI — brute-force search | **Dynamics-aware** compression (velocity, gradient, saliency) |
| Manual KD tuning | **Auto-diagnose** teacher-student gaps |

## Installation

```bash
pip install git+https://github.com/hw01931/FlowGrad.git

# With all optional dependencies
pip install "flowgrad[all] @ git+https://github.com/hw01931/FlowGrad.git"
```

## Quick Start

### 🔬 Deep Learning (PyTorch)

```python
from flowgrad import FlowTracker

tracker = FlowTracker(model, optimizer=optimizer, scheduler=scheduler, run_name="exp_01")

for epoch in range(100):
    loss = train_one_epoch(model, loader, optimizer)
    tracker.step(loss=loss.item())

tracker.report()                     # Text diagnostics with prescriptions
tracker.plot.full_report()           # 6-panel visual dashboard
tracker.export_for_agent()           # XML for AI assistants
```

### 🌲 Gradient Boosting

```python
from flowgrad import BoostingTracker
tracker = BoostingTracker()
model = xgb.train(params, dtrain,
    callbacks=[tracker.as_xgb_callback()])
tracker.report()
```

### 🔧 scikit-learn

```python
from flowgrad import SklearnTracker
tracker = SklearnTracker(feature_names=feature_names)
model = GradientBoostingClassifier(n_estimators=200, warm_start=True)
tracker.track_warm_start(model, X_train, y_train, X_val, y_val)
```

### 🧪 Feature Engineering + VIF

```python
from flowgrad import FeatureAnalyzer
analyzer = FeatureAnalyzer(model, X, y, feature_names=names)
suggestions = analyzer.suggest_features(top_k=10, collinearity_check=True, vif_threshold=10.0)
```

### 🗜️ Model Compression — Goal-Based Auto Search

```python
from flowgrad import CompressionTracker
tracker = CompressionTracker(model, eval_fn=lambda m: accuracy(m, X_val, y_val))
result = tracker.auto_compress(performance_floor=0.95, search_range=(0.1, 0.9))
tracker.layer_sensitivity(sparsity_levels=[0.1, 0.3, 0.5, 0.7, 0.9])
tracker.recommend_nonuniform(performance_floor=0.95)
```

### 🔬 Dynamic Saliency — Intelligent Pruning Priority

Unlike L1 pruning (weight magnitude), FlowGrad uses **training dynamics** to decide what to prune.

```python
from flowgrad import SaliencyAnalyzer
sa = SaliencyAnalyzer(tracker)
priority = sa.pruning_priority()
# → [("layer3.weight", 0.92, "Near-zero velocity; Declining gradient"), ...]
sa.report()
```

### 🔢 Quantization Advisor — Mixed-Precision Guidance

```python
from flowgrad import QuantizationAdvisor
qa = QuantizationAdvisor(tracker)
plan = qa.recommend_mixed_precision()
# → {"layer1.weight": 8, "layer2.weight": 4, "layer3.weight": 16}
qa.report()
```

### 🎓 Knowledge Distillation Tracker

```python
from flowgrad import DistillationTracker
dt = DistillationTracker(teacher_tracker, student_tracker)
gaps = dt.flow_gap()           # Where is the student struggling?
weights = dt.suggest_distillation_weights()  # Auto per-layer KD loss weights
dt.report()
```

### 🔌 LoRA / PEFT Tracker

```python
from flowgrad import PEFTTracker
pt = PEFTTracker(tracker)
ranks = pt.recommend_ranks()   # Per-layer LoRA rank recommendation
pt.adapter_utilization()       # Which adapters are actually learning?
pt.report()
```

### 🤖 AI Agent Mode — XML Output for Cursor/Copilot

```python
xml = tracker.export_for_agent()
print(xml)  # AI reads this and auto-applies fixes
```

```xml
<flowgrad_agent_report>
  <experiment_history> ... </experiment_history>
  <environment> <optimizer>AdamW(lr=0.001)</optimizer> </environment>
  <model_architecture> ... </model_architecture>
  <training_state> <current_loss>0.31</current_loss> </training_state>
  <diagnostics>
    <finding>
      <type>STAGNATION</type>
      <explanation><logic>Velocity < 1e-7 for 5+ steps</logic></explanation>
      <prescription><suggestion>Increase LR for this layer</suggestion></prescription>
    </finding>
  </diagnostics>
</flowgrad_agent_report>
```

## Features

### 🩺 Automated Diagnostics

```
⚠️  Alerts & Prescriptions
─────────────────────────────────────────────
  🧊 STAGNATION: 'layer3.conv.weight'
     Velocity ≈ 1.2e-09 since step 45
     💊 Increase learning rate or remove weight decay for this layer.

  💥 GRADIENT_EXPLOSION: 'layer1.fc.weight'
     Value: 512.3 at step 12
     💊 Add gradient clipping (max_norm=1.0) or reduce lr by 50%.
```

### What's tracked

| Domain | Metrics |
|---|---|
| **DL Layers** | Weight norm/mean/std, gradient norm/SNR, velocity, acceleration, dead neurons |
| **Boosting** | Per-round feature importance, train/valid metrics, overfitting gap |
| **Feature Eng** | Pairwise synergy, combination lift + VIF, redundancy, clustering |
| **Compression** | Sparsity, model size, layer sensitivity, Pareto frontier |
| **Saliency** | Velocity saliency, gradient momentum, pruning priority |
| **Quantization** | Weight range, SNR, per-layer bit-width recommendation |
| **Distillation** | Teacher-student velocity gap, SNR comparison, KD weights |
| **PEFT/LoRA** | Adapter utilization, rank recommendation, efficiency ratio |

## API Reference

| Class | Purpose | Input |
|---|---|---|
| `FlowTracker(model)` | DL training dynamics | PyTorch `nn.Module` |
| `BoostingTracker()` | Boosting round tracking | XGBoost/LightGBM/CatBoost |
| `SklearnTracker()` | sklearn model tracking | Any sklearn estimator |
| `FeatureAnalyzer(model, X, y)` | Feature engineering | Any fitted model + data |
| `CompressionTracker(model, eval_fn)` | Auto compression search | PyTorch model + eval |
| `SaliencyAnalyzer(tracker)` | Dynamic pruning priority | FlowTracker |
| `QuantizationAdvisor(tracker)` | Mixed-precision guidance | FlowTracker |
| `DistillationTracker(t, s)` | KD diagnostics | Two FlowTrackers |
| `PEFTTracker(tracker)` | LoRA/adapter analysis | FlowTracker |
| `AgentExporter` | AI-native XML output | Any tracker |

Every module exposes:
- `.report()` → text diagnostics
- `.to_agent_xml()` → structured XML for AI agents
- `.plot.*` → matplotlib figures (where applicable)

## Examples

📓 **[FlowGrad Demo Notebook](examples/flowgrad_demo.ipynb)** — All features on real data.

## Roadmap

- [x] v0.1 — DL Tracker + Boosting Tracker
- [x] v0.2 — scikit-learn support + Feature Engineering
- [x] v0.3 — Model Compression (auto-search, layer sensitivity, VIF)
- [x] v0.4 — AI Agent Mode (XML, history, explanation engine), Dynamic Saliency, Quantization Advisor, Distillation, LoRA/PEFT
- [ ] v0.5— RecSys module (embedding drift, coverage, cold start)
- [ ] v0.6 — Plotly interactive dashboards
- [ ] v1.0 — PyPI release

## License

[MIT](LICENSE)
