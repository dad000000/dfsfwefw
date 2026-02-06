from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class InactivityExplanation:
    inactive_reason: str
    top_factors: List[str]
    suggested_fix: str


class InactivityAnalyzer:
    """Diagnostics-only analyzer.

    IMPORTANT:
      - This module does not change any parameters.
      - It only explains why we are idle and suggests directions.
    """

    def explain(self, meta: Dict[str, object], metrics: Dict[str, object]) -> InactivityExplanation:
        no_reason = str(meta.get("no_trade_reason") or "")
        gate_state = str(meta.get("gate_state") or "")
        trading_enabled = bool(meta.get("trading_enabled", True))
        pause_reason = str(meta.get("pause_reason") or "")

        rr = float(metrics.get("reject_rate_5m") or 0.0)
        cr = float(metrics.get("cancel_rate_5m") or 0.0)
        fr = float(metrics.get("fill_rate_5m") or 0.0)
        evr = float(metrics.get("ev_gate_open_rate_5m") or 0.0)

        factors: List[str] = []
        fix = ""

        if not trading_enabled:
            factors.append("PAUSED")
            if pause_reason:
                factors.append(pause_reason)
            fix = "Wait until pause ends; reduce-only exits remain active."
            return InactivityExplanation("PAUSED", factors[:3], fix)

        if gate_state:
            factors.append(gate_state)
        if no_reason:
            factors.append(no_reason)

        # Heuristics for suggestions
        if "NO_SIGNAL" in no_reason or gate_state == "NO_SIGNAL":
            fix = "Signal is neutral/weak right now; wait for dir != 0 and edge > fees."
        elif "TOXIC" in no_reason or gate_state == "REGIME":
            fix = "Regime is toxic; wait for spreads/churn to normalize."
        elif gate_state == "EV" or "EV_GATE" in no_reason or "EV_LOW" in no_reason:
            fix = "Edge below fee-aware threshold; wait or relax min_edge_mult (carefully)."
        elif gate_state == "COOLDOWN" or "COOLDOWN" in no_reason:
            fix = "Cooldown active; bot is intentionally waiting to avoid churn."
        elif rr > 0.10:
            factors.append("REJECTS_HIGH")
            fix = "Post-only rejects high; consider larger offset / slower requotes."
        elif cr > 0.25:
            factors.append("CANCELS_HIGH")
            fix = "Cancel churn high; consider increasing cooldown / requote_min_ms."
        elif fr < 0.10 and evr > 0.20 and rr < 0.05:
            factors.append("FILL_RATE_LOW")
            fix = "EV gate opens but fills are rare; offset might be too large."
        else:
            fix = "Idle is explainable by gates/cooldowns; check execution metrics for hints."

        return InactivityExplanation(gate_state or "IDLE", factors[:3], fix)

    def meta(self, explanation: InactivityExplanation) -> Dict[str, object]:
        return {
            "inactive_reason": str(explanation.inactive_reason or ""),
            "inactive_top_factors": list(explanation.top_factors or []),
            "inactive_suggested_fix": str(explanation.suggested_fix or ""),
        }

