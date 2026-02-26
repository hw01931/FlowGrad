"""
FlowGrad Agent Mode — Structured XML export for AI coding assistants.

Generates diagnostic output optimized for AI agents (Cursor, Copilot, Antigravity)
that can read terminal output and automatically apply fixes.

The XML includes:
  1. Experiment history (cross-run comparison)
  2. Environment context (optimizer, scheduler, architecture)
  3. Training state summary (loss trend, current LR)
  4. Diagnostics with math/logic explanations and prescriptions
"""
from __future__ import annotations

import textwrap
from typing import Any, Dict, List, Optional
from xml.sax.saxutils import escape

from flowgrad.history import HistoryTracker


# ======================================================================
#  Explanation Engine — Formulas & Logic for each diagnostic type
# ======================================================================

EXPLANATIONS = {
    "STAGNATION": {
        "logic": "Velocity = ||W(t) - W(t-1)||_2;  Alert when Velocity < 1e-7 for 5+ consecutive steps",
        "description": (
            "This layer's weights are barely changing. The gradient signal is too weak "
            "to update parameters meaningfully at the current learning rate."
        ),
        "action": "MODIFY_HYPERPARAMETER",
        "suggestion": (
            "Increase learning rate for this layer (use per-param-group lr), "
            "remove or reduce weight_decay, or check for vanishing gradients upstream. "
            "This layer is also a strong candidate for aggressive pruning."
        ),
    },
    "GRADIENT_EXPLOSION": {
        "logic": "||grad||_2 > 100.0  (L2 norm of gradient tensor)",
        "description": (
            "The gradient magnitude is extremely large, causing the optimizer to take "
            "oversized steps that destabilize training."
        ),
        "action": "ADD_CODE",
        "suggestion": (
            "Add `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` "
            "before `optimizer.step()`. Alternatively, reduce learning rate by 50%."
        ),
    },
    "VELOCITY_EXPLOSION": {
        "logic": "||W(t) - W(t-1)||_2 > 100.0  (weight change per step)",
        "description": (
            "Weights are changing too rapidly, indicating numerical instability "
            "or an excessively high learning rate."
        ),
        "action": "MODIFY_HYPERPARAMETER",
        "suggestion": "Reduce learning rate by 2-5x. Check for NaN in loss values.",
    },
    "DEAD_NEURONS": {
        "logic": "Dead Ratio = count(|weight| < 1e-6) / total_params;  Alert when ratio > 50%",
        "description": (
            "Over half the parameters in this layer are effectively zero. "
            "With standard ReLU, neurons that output zero for all inputs stop receiving gradients."
        ),
        "action": "MODIFY_ARCHITECTURE",
        "suggestion": (
            "Replace `nn.ReLU()` with `nn.LeakyReLU(0.01)` or `nn.GELU()`. "
            "Also consider Kaiming He initialization: `nn.init.kaiming_normal_(layer.weight)`."
        ),
    },
    "LOW_GRADIENT_SNR": {
        "logic": "SNR = mean(grad)^2 / var(grad);  Alert when SNR < 0.01",
        "description": (
            "The gradient signal is noisy relative to its magnitude. "
            "The optimizer is essentially taking random walks instead of descending."
        ),
        "action": "MODIFY_HYPERPARAMETER",
        "suggestion": (
            "Increase batch size to reduce gradient noise, or reduce learning rate. "
            "If using data augmentation, verify it's not too aggressive."
        ),
    },
}


# ======================================================================
#  Context Scanner — extracts optimizer, scheduler, architecture info
# ======================================================================

def _scan_optimizer(tracker) -> str:
    """Extract optimizer info as a clean string."""
    opt = getattr(tracker, "optimizer", None)
    if opt is None:
        return "Not provided"
    name = type(opt).__name__
    pg = opt.param_groups[0]
    lr = pg.get("lr", "?")
    wd = pg.get("weight_decay", 0)
    betas = pg.get("betas", None)
    parts = [f"lr={lr}", f"weight_decay={wd}"]
    if betas:
        parts.append(f"betas={betas}")
    return f"{name}({', '.join(parts)})"


def _scan_scheduler(tracker) -> str:
    """Extract LR scheduler info."""
    sched = getattr(tracker, "scheduler", None)
    if sched is None:
        return "None"
    name = type(sched).__name__
    try:
        current_lr = sched.get_last_lr()[0]
        return f"{name}(current_lr={current_lr})"
    except Exception:
        return name


def _scan_architecture(model) -> Dict[str, Any]:
    """Scan model for structural summary."""
    total_params = 0
    trainable_params = 0
    norm_layers = 0
    dropout_layers = 0
    activation_types = set()

    for m in model.modules():
        cls_name = type(m).__name__.lower()
        if "norm" in cls_name:
            norm_layers += 1
        if "dropout" in cls_name:
            dropout_layers += 1
        if any(act in cls_name for act in ["relu", "gelu", "silu", "tanh", "sigmoid", "leaky"]):
            activation_types.add(type(m).__name__)

    for p in model.parameters():
        total_params += p.numel()
        if p.requires_grad:
            trainable_params += p.numel()

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "normalization_layers": norm_layers,
        "dropout_layers": dropout_layers,
        "activations": sorted(activation_types) if activation_types else ["None detected"],
    }


# ======================================================================
#  XML Builder Helpers
# ======================================================================

def _tag(name: str, content: str, indent: int = 2) -> str:
    """Multi-line XML tag."""
    sp = " " * indent
    inner = textwrap.indent(content.strip(), sp + "  ")
    return f"{sp}<{name}>\n{inner}\n{sp}</{name}>"


def _line(name: str, value, indent: int = 2) -> str:
    """Single-line XML tag."""
    sp = " " * indent
    return f"{sp}<{name}>{escape(str(value))}</{name}>"


# ======================================================================
#  Agent Exporter
# ======================================================================

class AgentExporter:
    """
    Exports FlowGrad diagnostics into structured XML for AI agents.

    Usage:
        tracker = FlowTracker(model, optimizer=opt, scheduler=sched, run_name="exp_03")
        # ... training ...
        xml = tracker.export_for_agent()
        print(xml)  # AI reads this from terminal
    """

    @classmethod
    def export_dl(
        cls,
        tracker,
        run_name: str = "current_run",
        include_history: bool = True,
        save: bool = True,
    ) -> str:
        """
        Export DL training diagnostics as structured XML.

        Args:
            tracker: FlowTracker instance.
            run_name: Identifier for this experiment run.
            include_history: Include previous runs from .flowgrad/history.jsonl.
            save: Save this run's summary to history.

        Returns:
            XML string for AI agent consumption.
        """
        from flowgrad.analyzers.velocity import detect_stagnation, detect_explosion
        from flowgrad.analyzers.health import layer_health_score, gradient_snr_per_layer

        store = tracker.store

        # ── 1. Environment ──────────────────────────────────────────
        opt_str = _scan_optimizer(tracker)
        sched_str = _scan_scheduler(tracker)
        arch = _scan_architecture(tracker.model)

        # ── 2. Training State ───────────────────────────────────────
        losses = store.get_loss_history()
        valid_losses = [l for l in losses if l is not None]

        # ── 3. Diagnostics ──────────────────────────────────────────
        findings: List[Dict[str, str]] = []
        issue_summaries: List[str] = []

        # Stagnation
        for alert in detect_stagnation(store):
            key = "STAGNATION"
            exp = EXPLANATIONS[key]
            findings.append({
                "type": key,
                "location": alert["name"],
                "value": f"velocity={alert['current_velocity']:.2e}, stagnant since step {alert['stagnant_since_step']}",
                "logic": exp["logic"],
                "description": exp["description"],
                "action": exp["action"],
                "suggestion": exp["suggestion"],
            })
            issue_summaries.append(f"{key} in {alert['name']}")

        # Gradient / velocity explosion
        seen_explosions = set()
        for alert in detect_explosion(store):
            key_name = (alert["name"], alert["type"])
            if key_name in seen_explosions:
                continue
            seen_explosions.add(key_name)
            exp_key = alert["type"].upper().replace("_", "_")
            if exp_key == "GRADIENT_EXPLOSION":
                exp = EXPLANATIONS["GRADIENT_EXPLOSION"]
            else:
                exp = EXPLANATIONS["VELOCITY_EXPLOSION"]
            findings.append({
                "type": exp_key,
                "location": alert["name"],
                "value": f"{alert['value']:.2e} at step {alert['step']}",
                "logic": exp["logic"],
                "description": exp["description"],
                "action": exp["action"],
                "suggestion": exp["suggestion"],
            })
            issue_summaries.append(f"{exp_key} in {alert['name']}")

        # Dead neurons
        for name in store.layer_names:
            history = store.get_layer_history(name)
            if history and history[-1].dead_ratio > 0.5:
                exp = EXPLANATIONS["DEAD_NEURONS"]
                findings.append({
                    "type": "DEAD_NEURONS",
                    "location": name,
                    "value": f"{history[-1].dead_ratio * 100:.1f}% near-zero",
                    "logic": exp["logic"],
                    "description": exp["description"],
                    "action": exp["action"],
                    "suggestion": exp["suggestion"],
                })
                issue_summaries.append(f"DEAD_NEURONS in {name}")

        # Low gradient SNR
        snr_data = gradient_snr_per_layer(store)
        for name, series in snr_data.items():
            if series and 0 < series[-1] < 0.01:
                exp = EXPLANATIONS["LOW_GRADIENT_SNR"]
                findings.append({
                    "type": "LOW_GRADIENT_SNR",
                    "location": name,
                    "value": f"SNR={series[-1]:.4e}",
                    "logic": exp["logic"],
                    "description": exp["description"],
                    "action": exp["action"],
                    "suggestion": exp["suggestion"],
                })
                issue_summaries.append(f"LOW_SNR in {name}")

        # Health scores
        health = layer_health_score(store)

        # ── 4. Build Run Data for History ───────────────────────────
        run_data = {
            "run_id": run_name,
            "optimizer": opt_str,
            "total_params": f"{arch['total_params']/1e6:.2f}M",
            "steps": store.num_steps,
            "final_loss": round(valid_losses[-1], 4) if valid_losses else None,
            "issues": issue_summaries,
            "avg_health": round(sum(health.values()) / max(len(health), 1), 1),
        }

        if save:
            HistoryTracker.append_run(run_data)

        # ── 5. Assemble XML ─────────────────────────────────────────
        xml_parts = ['<flowgrad_agent_report>']

        # A. History
        if include_history:
            past = HistoryTracker.get_recent_runs(n=5)
            past = [r for r in past if r.get("run_id") != run_name]
            if past:
                hist_lines = []
                for r in past[-3:]:
                    issues_str = ", ".join(r.get("issues", [])) or "None"
                    hist_lines.append(
                        _tag("run", "\n".join([
                            _line("id", r.get("run_id", "?"), 0),
                            _line("timestamp", r.get("timestamp", "?"), 0),
                            _line("optimizer", r.get("optimizer", "?"), 0),
                            _line("steps", r.get("steps", "?"), 0),
                            _line("final_loss", r.get("final_loss", "?"), 0),
                            _line("issues", issues_str, 0),
                            _line("avg_health", r.get("avg_health", "?"), 0),
                        ]), indent=4)
                    )
                xml_parts.append(_tag("experiment_history", "\n".join(hist_lines)))

        # B. Environment
        env = "\n".join([
            _line("optimizer", opt_str, 4),
            _line("lr_scheduler", sched_str, 4),
        ])
        xml_parts.append(_tag("environment", env))

        # C. Architecture
        arch_lines = "\n".join([
            _line("total_params", f"{arch['total_params']/1e6:.2f}M", 4),
            _line("trainable_params", f"{arch['trainable_params']/1e6:.2f}M", 4),
            _line("normalization_layers", arch["normalization_layers"], 4),
            _line("dropout_layers", arch["dropout_layers"], 4),
            _line("activations", ", ".join(arch["activations"]), 4),
        ])
        xml_parts.append(_tag("model_architecture", arch_lines))

        # D. Training State
        state_lines = [_line("current_step", store.num_steps, 4)]
        if valid_losses:
            state_lines.append(_line("initial_loss", f"{valid_losses[0]:.4f}", 4))
            state_lines.append(_line("current_loss", f"{valid_losses[-1]:.4f}", 4))
            if len(valid_losses) > 1:
                state_lines.append(_line("loss_trend", f"{valid_losses[-1] - valid_losses[0]:+.4f}", 4))
                state_lines.append(_line("min_loss", f"{min(valid_losses):.4f}", 4))
        xml_parts.append(_tag("training_state", "\n".join(state_lines)))

        # E. Diagnostics
        if findings:
            finding_xmls = []
            for f in findings:
                finding_xml = "\n".join([
                    _line("type", f["type"], 6),
                    _line("location", f["location"], 6),
                    _line("value", f["value"], 6),
                    _tag("explanation", "\n".join([
                        _line("logic", f["logic"], 0),
                        _line("description", f["description"], 0),
                    ]), indent=6),
                    _tag("prescription", "\n".join([
                        _line("action", f["action"], 0),
                        _line("suggestion", f["suggestion"], 0),
                    ]), indent=6),
                ])
                finding_xmls.append(_tag("finding", finding_xml, indent=4))
            xml_parts.append(_tag("diagnostics", "\n".join(finding_xmls)))
        else:
            xml_parts.append(_tag("diagnostics",
                _line("status", "HEALTHY — No critical issues detected.", 4)))

        # F. Layer Health Summary
        health_lines = []
        for name in sorted(health, key=lambda k: health[k]):
            score = health[name]
            status = "CRITICAL" if score < 40 else "WARNING" if score < 70 else "HEALTHY"
            health_lines.append(
                f'    <layer name="{escape(name)}" health="{score:.0f}" status="{status}" />'
            )
        xml_parts.append(_tag("layer_health_summary", "\n".join(health_lines)))

        xml_parts.append("</flowgrad_agent_report>")
        return "\n\n".join(xml_parts)
