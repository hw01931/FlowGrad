"""Tests for FlowTracker (DL core)."""
import pytest
import numpy as np


def _make_simple_model():
    """Create a simple model for testing."""
    import torch
    import torch.nn as nn
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5),
    )
    return model


@pytest.fixture
def model_and_tracker():
    """Fixture: simple model + tracker."""
    import torch
    from flowgrad import FlowTracker

    model = _make_simple_model()
    tracker = FlowTracker(model)
    return model, tracker


class TestFlowTrackerInit:
    def test_creates_tracker(self, model_and_tracker):
        model, tracker = model_and_tracker
        assert tracker is not None
        assert len(tracker._param_names) > 0

    def test_excludes_bias_by_default(self, model_and_tracker):
        _, tracker = model_and_tracker
        for name in tracker._param_names:
            assert "bias" not in name

    def test_includes_bias_when_requested(self):
        import torch
        from flowgrad import FlowTracker
        model = _make_simple_model()
        tracker = FlowTracker(model, include_bias=True)
        has_bias = any("bias" in n for n in tracker._param_names)
        assert has_bias

    def test_repr(self, model_and_tracker):
        _, tracker = model_and_tracker
        r = repr(tracker)
        assert "FlowTracker" in r
        assert "layers=" in r


class TestFlowTrackerStep:
    def test_step_increments(self, model_and_tracker):
        model, tracker = model_and_tracker
        import torch

        # Do a fake forward/backward
        x = torch.randn(4, 10)
        loss = model(x).sum()
        loss.backward()

        tracker.step(loss=loss.item())
        assert tracker.store.num_steps == 1
        assert tracker._step_count == 1

    def test_step_records_loss(self, model_and_tracker):
        model, tracker = model_and_tracker
        import torch

        x = torch.randn(4, 10)
        loss = model(x).sum()
        loss.backward()
        tracker.step(loss=42.0)

        assert tracker.store.get_loss_history() == [42.0]

    def test_step_records_weight_stats(self, model_and_tracker):
        model, tracker = model_and_tracker
        import torch

        x = torch.randn(4, 10)
        loss = model(x).sum()
        loss.backward()
        tracker.step(loss=loss.item())

        for name in tracker._param_names:
            snap = tracker.store.steps[0].layers[name]
            assert snap.weight_norm > 0
            assert snap.num_params > 0

    def test_step_records_grad_stats(self, model_and_tracker):
        model, tracker = model_and_tracker
        import torch

        x = torch.randn(4, 10)
        loss = model(x).sum()
        loss.backward()
        tracker.step(loss=loss.item())

        for name in tracker._param_names:
            snap = tracker.store.steps[0].layers[name]
            assert snap.grad_norm >= 0

    def test_velocity_computed_after_2_steps(self, model_and_tracker):
        model, tracker = model_and_tracker
        import torch
        import torch.optim as optim

        opt = optim.SGD(model.parameters(), lr=0.01)

        for _ in range(3):
            opt.zero_grad()
            x = torch.randn(4, 10)
            loss = model(x).sum()
            loss.backward()
            opt.step()
            tracker.step(loss=loss.item())

        # After step 2+, velocity should be > 0
        first_layer = tracker._param_names[0]
        snap2 = tracker.store.steps[1].layers[first_layer]
        assert snap2.velocity > 0

    def test_acceleration_computed_after_3_steps(self, model_and_tracker):
        model, tracker = model_and_tracker
        import torch
        import torch.optim as optim

        opt = optim.SGD(model.parameters(), lr=0.01)

        for _ in range(3):
            opt.zero_grad()
            x = torch.randn(4, 10)
            loss = model(x).sum()
            loss.backward()
            opt.step()
            tracker.step(loss=loss.item())

        first_layer = tracker._param_names[0]
        snap3 = tracker.store.steps[2].layers[first_layer]
        assert snap3.acceleration >= 0


class TestFlowTrackerSummary:
    def test_summary_keys(self, model_and_tracker):
        model, tracker = model_and_tracker
        import torch

        x = torch.randn(4, 10)
        loss = model(x).sum()
        loss.backward()
        tracker.step(loss=loss.item())

        s = tracker.summary
        assert "total_steps" in s
        assert "num_layers" in s
        assert "layer_names" in s
        assert s["total_steps"] == 1


class TestFlowTrackerDetach:
    def test_detach_removes_hooks(self, model_and_tracker):
        _, tracker = model_and_tracker
        assert len(tracker._hooks) > 0
        tracker.detach()
        assert len(tracker._hooks) == 0
