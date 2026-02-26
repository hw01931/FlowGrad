# FlowGrad 🌊

**ML 학습 과정 진단 + 피처 엔지니어링 라이브러리** — 코드 한 줄로 모델 학습 역학을 추적하고, 피처 상호작용을 분석합니다.

## ✨ Features

- 🔬 **PyTorch**: 레이어별 가중치 속도·가속도·건강 상태 자동 추적
- 🌲 **XGBoost / LightGBM / CatBoost**: 라운드별 피처 중요도 변화·과적합 탐지
- � **scikit-learn**: GradientBoosting(warm_start), RandomForest(per-tree), SGD(partial_fit) 지원
- 🧪 **Feature Engineering**: 피처 상호작용·조합 제안·중복 탐지·클러스터링
- �📊 **시각화**: 다크 테마 대시보드, 히트맵, SNR 차트 등 15+ 차트
- 💊 **자동 진단**: 정체·폭주·과적합 탐지 + 텍스트 처방

## Quick Start

### Installation

```bash
pip install -e ".[all]"
```

### 1. PyTorch — DL Training Tracker

```python
from flowgrad import FlowTracker

tracker = FlowTracker(model)

for epoch in range(100):
    loss = train_one_epoch(model, loader, optimizer)
    tracker.step(loss=loss.item())

tracker.report()                  # 종합 진단 리포트
tracker.plot.velocity_heatmap()   # 레이어별 학습 속도 히트맵
tracker.plot.health_dashboard()   # 레이어 건강 상태
tracker.plot.full_report()        # 종합 대시보드 (6개 차트)
```

### 2. XGBoost / LightGBM / CatBoost

```python
from flowgrad import BoostingTracker
import xgboost as xgb

tracker = BoostingTracker()
model = xgb.train(params, dtrain, num_boost_round=500,
                  evals=[(dtrain, "train"), (dvalid, "valid")],
                  callbacks=[tracker.as_xgb_callback()])

tracker.report()
tracker.plot.feature_drift()              # 피처 중요도 변화
tracker.plot.overfitting_detector()       # 과적합 탐지
```

```python
# LightGBM
tracker = BoostingTracker()
model = lgb.train(params, dtrain, callbacks=[tracker.as_lgb_callback()])

# CatBoost
tracker = BoostingTracker()
model = CatBoostClassifier(iterations=500)
model.fit(X, y, callbacks=[tracker.as_catboost_callback()])
```

### 3. scikit-learn

```python
from flowgrad import SklearnTracker
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

# GradientBoosting (warm_start 방식으로 라운드별 추적)
tracker = SklearnTracker(feature_names=feature_names)
model = GradientBoostingClassifier(n_estimators=200, warm_start=True)
tracker.track_warm_start(model, X_train, y_train, X_val, y_val, step_size=10)
tracker.report()

# RandomForest (개별 트리 분석)
model = RandomForestClassifier(n_estimators=100).fit(X, y)
tracker = SklearnTracker.from_forest(model, feature_names=feature_names)
tracker.plot.feature_drift()  # 트리별 피처 중요도 변화

# SGDClassifier (partial_fit 배치별 추적)
tracker = SklearnTracker()
tracker.track_partial_fit(model, X_batches, y_batches, classes=[0, 1])
```

### 4. Feature Engineering ⭐ (차별화 기능)

```python
from flowgrad import FeatureAnalyzer

analyzer = FeatureAnalyzer(model, X_train, y_train, feature_names=feature_names)

# 피처 상호작용 분석 (기존 corr()과 다르게 비선형 시너지 측정)
interactions = analyzer.interactions(top_k=10)
# → [{"feat_a": "age", "feat_b": "income", "synergy_score": 0.12}, ...]

# 피처 조합 제안 (A*B, A/B 등 자동 테스트)
suggestions = analyzer.suggest_features(top_k=10)
# → [{"expression": "age * income", "lift": 0.08, "target_correlation": 0.72}, ...]

# 중복 피처 탐지
redundant = analyzer.redundant_features(threshold=0.95)
# → [{"feat_a": "height_cm", "feat_b": "height_inch", "recommendation": "Drop height_inch"}]

# 피처 클러스터링
clusters = analyzer.feature_clusters()
# → [{"cluster_id": 0, "features": ["age", "income"], "cohesion": 0.85}, ...]

# 종합 리포트
analyzer.report()

# 시각화
analyzer.plot.interaction_heatmap()   # 상호작용 히트맵
analyzer.plot.suggestion_chart()      # 조합 제안 차트
analyzer.plot.redundancy_graph()      # 중복 네트워크
analyzer.plot.cluster_map()           # 클러스터 맵
```

## 기존 도구와의 차이점

| 기존 | FlowGrad |
|---|---|
| `df.corr()` | 선형 상관만 | **비선형 상호작용 + 시너지** 측정 |
| `model.feature_importances_` | 학습 끝난 후 결과론적 | **학습 중** 실시간 변화 추적 |
| SHAP | "왜 이 예측?" (결과 해석) | **어떤 피처를 만들면 좋을지** 제안 |
| TensorBoard | 수동 로깅 필요 | **한 줄**이면 전체 추적 시작 |

## Available Plots

### DL (PyTorch) — 7 charts
`loss()` · `velocity_heatmap()` · `gradient_flow()` · `weight_distribution()` · `health_dashboard()` · `gradient_snr()` · `full_report()`

### Boosting / sklearn — 5 charts
`eval_metrics()` · `feature_drift()` · `feature_importance_heatmap()` · `overfitting_detector()` · `full_report()`

### Feature Engineering — 4 charts
`interaction_heatmap()` · `suggestion_chart()` · `redundancy_graph()` · `cluster_map()`

## License

MIT
