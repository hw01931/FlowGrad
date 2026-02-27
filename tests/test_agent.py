"""Tests for Agent Mode: HistoryTracker + AgentExporter XML generation."""
import os
import json


def test_history_tracker_append_and_get():
    """Test JSONL history append and retrieval."""
    from gradtracer.history import HistoryTracker

    path = HistoryTracker._get_path()
    if os.path.exists(path):
        os.remove(path)

    HistoryTracker.append_run({"run_id": "exp_01", "loss": 0.5})
    HistoryTracker.append_run({"run_id": "exp_02", "loss": 0.3})

    runs = HistoryTracker.get_recent_runs(5)
    assert len(runs) == 2
    assert runs[0]["run_id"] == "exp_01"
    assert runs[1]["run_id"] == "exp_02"
    assert "timestamp" in runs[0]

    # Cleanup
    if os.path.exists(path):
        os.remove(path)


def test_history_empty():
    """Test empty history."""
    from gradtracer.history import HistoryTracker

    path = HistoryTracker._get_path()
    if os.path.exists(path):
        os.remove(path)

    runs = HistoryTracker.get_recent_runs(5)
    assert runs == []


def test_agent_xml_structure():
    """Test XML output structure with optimizer and scheduler context."""
    import torch
    import torch.nn as nn
    from gradtracer import FlowTracker

    model = nn.Sequential(
        nn.Linear(10, 32),
        nn.BatchNorm1d(32),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(32, 1),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)

    tracker = FlowTracker(
        model, optimizer=optimizer, scheduler=scheduler, run_name="test_xml_01"
    )

    # Run a few training steps
    X = torch.randn(32, 10)
    for _ in range(5):
        out = model(X)
        loss = out.sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        tracker.step(loss=loss.item())

    xml = tracker.export_for_agent(include_history=False, save=False)

    # Verify all XML sections
    assert "<gradtracer_agent_report>" in xml
    assert "<environment>" in xml
    assert "Adam" in xml
    assert "StepLR" in xml
    assert "<model_architecture>" in xml
    assert "Normalization" in xml or "normalization" in xml
    assert "Dropout" in xml or "dropout" in xml
    assert "<training_state>" in xml
    assert "<current_step>5</current_step>" in xml
    assert "<diagnostics>" in xml
    assert "<layer_health_summary>" in xml
    assert "</gradtracer_agent_report>" in xml


def test_agent_xml_with_history():
    """Test that previous runs appear in experiment_history."""
    import torch
    import torch.nn as nn
    from gradtracer import FlowTracker
    from gradtracer.history import HistoryTracker

    path = HistoryTracker._get_path()
    if os.path.exists(path):
        os.remove(path)

    # Simulate a past run
    HistoryTracker.append_run({
        "run_id": "past_exp_baseline",
        "optimizer": "SGD(lr=0.1)",
        "issues": ["GRADIENT_EXPLOSION in layer1"],
        "final_loss": 1.5,
        "steps": 100,
    })

    model = nn.Linear(10, 1)
    tracker = FlowTracker(model, run_name="current_exp")
    tracker.step(loss=0.5)

    xml = tracker.export_for_agent(include_history=True, save=False)

    assert "<experiment_history>" in xml
    assert "past_exp_baseline" in xml
    assert "SGD(lr=0.1)" in xml

    # Cleanup
    if os.path.exists(path):
        os.remove(path)


def test_agent_explanation_engine():
    """Test that findings include explanation tags with logic and description."""
    import torch
    import torch.nn as nn
    from gradtracer import FlowTracker

    # Create a model that will trigger dead neurons
    model = nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 1),
    )

    # Zero out weights to trigger dead neuron detection
    with torch.no_grad():
        model[0].weight.fill_(0.0)

    tracker = FlowTracker(model, run_name="dead_neuron_test")

    X = torch.randn(16, 10)
    for _ in range(6):
        out = model(X)
        loss = out.sum()
        loss.backward()
        tracker.step(loss=loss.item())

    xml = tracker.export_for_agent(include_history=False, save=False)

    # If dead neurons detected, verify explanation tags are present
    if "DEAD_NEURONS" in xml:
        assert "<explanation>" in xml
        assert "<logic>" in xml
        assert "<description>" in xml
        assert "<prescription>" in xml
        assert "<suggestion>" in xml


def test_agent_no_optimizer():
    """Test that XML works even without optimizer/scheduler."""
    import torch
    import torch.nn as nn
    from gradtracer import FlowTracker

    model = nn.Linear(5, 1)
    tracker = FlowTracker(model, run_name="minimal_test")
    tracker.step(loss=1.0)

    xml = tracker.export_for_agent(include_history=False, save=False)

    assert "<gradtracer_agent_report>" in xml
    assert "Not provided" in xml  # optimizer not given
    assert "None" in xml  # scheduler not given
