"""Tests for v0.5 analyzers: Saliency, Quantization, Distillation, PEFT."""
import torch
import torch.nn as nn
from gradtracer import FlowTracker, SaliencyAnalyzer, QuantizationAdvisor, DistillationTracker, PEFTTracker


def _train_tracker(model, steps=10):
    """Helper: attach a FlowTracker and run some steps."""
    tracker = FlowTracker(model, run_name="test_run")
    X = torch.randn(32, 10)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(steps):
        out = model(X).sum()
        out.backward()
        opt.step()
        opt.zero_grad()
        tracker.step(loss=out.item())
    return tracker


# ─── Saliency ───────────────────────────────────────────────

def test_saliency_velocity():
    model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 1))
    tracker = _train_tracker(model)
    sa = SaliencyAnalyzer(tracker)
    scores = sa.velocity_saliency()
    assert len(scores) > 0
    assert all(0 <= v <= 1.0 for v in scores.values())


def test_saliency_gradient_momentum():
    model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 1))
    tracker = _train_tracker(model)
    sa = SaliencyAnalyzer(tracker)
    momentum = sa.gradient_momentum()
    assert len(momentum) > 0


def test_saliency_pruning_priority():
    model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 1))
    tracker = _train_tracker(model)
    sa = SaliencyAnalyzer(tracker)
    priority = sa.pruning_priority()
    assert len(priority) > 0
    assert all(len(item) == 3 for item in priority)
    # Sorted descending by score
    scores = [p[1] for p in priority]
    assert scores == sorted(scores, reverse=True)


def test_saliency_report():
    model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 1))
    tracker = _train_tracker(model)
    sa = SaliencyAnalyzer(tracker)
    report = sa.report()
    assert "Saliency" in report


def test_saliency_xml():
    model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 1))
    tracker = _train_tracker(model)
    sa = SaliencyAnalyzer(tracker)
    xml = sa.to_agent_xml()
    assert "<saliency_analysis>" in xml


# ─── Quantization ──────────────────────────────────────────

def test_quantization_profile():
    model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 1))
    tracker = _train_tracker(model)
    qa = QuantizationAdvisor(tracker)
    profile = qa.sensitivity_profile()
    assert len(profile) > 0
    for name, info in profile.items():
        assert "recommended_bits" in info
        assert info["recommended_bits"] in (2, 4, 8, 16, 32)


def test_quantization_mixed_precision():
    model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 1))
    tracker = _train_tracker(model)
    qa = QuantizationAdvisor(tracker)
    plan = qa.recommend_mixed_precision()
    assert len(plan) > 0
    assert all(isinstance(bits, int) for bits in plan.values())


def test_quantization_size_reduction():
    model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 1))
    tracker = _train_tracker(model)
    qa = QuantizationAdvisor(tracker)
    reduction = qa.estimated_size_reduction()
    assert "original_bits" in reduction
    assert "avg_bits" in reduction
    assert "estimated_reduction_pct" in reduction


def test_quantization_report():
    model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 1))
    tracker = _train_tracker(model)
    qa = QuantizationAdvisor(tracker)
    report = qa.report()
    assert "Quantization" in report


def test_quantization_xml():
    model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 1))
    tracker = _train_tracker(model)
    qa = QuantizationAdvisor(tracker)
    xml = qa.to_agent_xml()
    assert "<quantization_analysis>" in xml


# ─── Distillation ──────────────────────────────────────────

def test_distillation_flow_gap():
    teacher = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 1))
    student = nn.Sequential(nn.Linear(10, 8), nn.ReLU(), nn.Linear(8, 1))
    t_tracker = _train_tracker(teacher)
    s_tracker = _train_tracker(student)
    dt = DistillationTracker(t_tracker, s_tracker)
    gaps = dt.flow_gap()
    assert len(gaps) > 0
    for name, info in gaps.items():
        assert "gap_ratio" in info
        assert "status" in info
        assert info["status"] in ("OK", "STRUGGLING", "CRITICAL")


def test_distillation_kd_weights():
    teacher = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 1))
    student = nn.Sequential(nn.Linear(10, 8), nn.ReLU(), nn.Linear(8, 1))
    t_tracker = _train_tracker(teacher)
    s_tracker = _train_tracker(student)
    dt = DistillationTracker(t_tracker, s_tracker)
    weights = dt.suggest_distillation_weights()
    assert len(weights) > 0
    assert abs(sum(weights.values()) - 1.0) < 0.01  # softmax sums to ~1


def test_distillation_report():
    teacher = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 1))
    student = nn.Sequential(nn.Linear(10, 8), nn.ReLU(), nn.Linear(8, 1))
    t_tracker = _train_tracker(teacher)
    s_tracker = _train_tracker(student)
    dt = DistillationTracker(t_tracker, s_tracker)
    report = dt.report()
    assert "Distillation" in report


def test_distillation_xml():
    teacher = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 1))
    student = nn.Sequential(nn.Linear(10, 8), nn.ReLU(), nn.Linear(8, 1))
    t_tracker = _train_tracker(teacher)
    s_tracker = _train_tracker(student)
    dt = DistillationTracker(t_tracker, s_tracker)
    xml = dt.to_agent_xml()
    assert "<distillation_analysis>" in xml


# ─── PEFT ──────────────────────────────────────────────────

def test_peft_rank_recommendation():
    model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 1))
    tracker = _train_tracker(model)
    pt = PEFTTracker(tracker)
    ranks = pt.recommend_ranks()
    assert len(ranks) > 0
    assert all(isinstance(r, int) for r in ranks.values())


def test_peft_rank_budget():
    model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 1))
    tracker = _train_tracker(model)
    pt = PEFTTracker(tracker)
    ranks = pt.recommend_ranks(budget=10)
    assert sum(ranks.values()) <= 12  # budget + rounding tolerance


def test_peft_analysis():
    model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 1))
    tracker = _train_tracker(model)
    pt = PEFTTracker(tracker)
    analysis = pt.adapter_vs_base_analysis()
    assert "recommendation" in analysis


def test_peft_report():
    model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 1))
    tracker = _train_tracker(model)
    pt = PEFTTracker(tracker)
    report = pt.report()
    assert "PEFT" in report or "LoRA" in report


def test_peft_xml():
    model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 1))
    tracker = _train_tracker(model)
    pt = PEFTTracker(tracker)
    xml = pt.to_agent_xml()
    assert "<peft_analysis>" in xml
