from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class ExecMetrics:
    """Rolling execution metrics snapshot used for bounded autotuning."""

    entry_attempts_5m: int = 0
    entry_fills_5m: int = 0
    fill_rate_5m: float = 0.0
    rejects_5m: int = 0
    cancels_5m: int = 0
    reject_rate_5m: float = 0.0
    cancel_rate_5m: float = 0.0
    time_to_fill_p50_ms: int = 0
    time_to_fill_p90_ms: int = 0
    time_to_fill_n: int = 0
    ev_gate_open_rate_5m: float = 0.0
    pnl_1h_after_fees: float = 0.0
    fees_1h: float = 0.0


@dataclass
class LiveParams:
    """Live, effective execution params (bounded adjustments only)."""

    offset_adj_ticks: int
    entry_cooldown_ms: int
    requote_min_ms: int
    min_edge_mult_live: float
    trading_enabled: bool
    pause_until_ms: int
    pause_reason: str


class Autotuner:
    """Minimal, bounded execution autotuner for swing entries.

    Design goals:
      - Only touches execution knobs (offset/cooldown/requote). No signal changes.
      - Fully observable via strategy_meta.
      - Bounded, slow, and safe (no rapid oscillations).
    """

    def __init__(self, s) -> None:
        # Hard caps
        self.enabled = bool(getattr(s, "AUTOTUNE_ENABLED", True))
        self.interval_ms = int(getattr(s, "AUTOTUNE_INTERVAL_MS", 30_000))
        self.off_min = int(getattr(s, "AUTOTUNE_OFFSET_ADJ_MIN", -6))
        self.off_max = int(getattr(s, "AUTOTUNE_OFFSET_ADJ_MAX", 12))
        self.cd_min = int(getattr(s, "AUTOTUNE_COOLDOWN_MIN_MS", 5_000))
        self.cd_max = int(getattr(s, "AUTOTUNE_COOLDOWN_MAX_MS", 30_000))
        self.rq_min_min = int(getattr(s, "AUTOTUNE_REQUOTE_MIN_MS_MIN", 3_000))
        self.rq_min_max = int(getattr(s, "AUTOTUNE_REQUOTE_MIN_MS_MAX", 12_000))

        # Thresholds (can be overridden by config)
        self.reject_rate_hi = float(getattr(s, "AUTOTUNE_REJECT_RATE_HI", 0.10))
        self.cancel_rate_hi = float(getattr(s, "AUTOTUNE_CANCEL_RATE_HI", 0.25))
        self.fill_rate_lo = float(getattr(s, "AUTOTUNE_FILL_RATE_LO", 0.15))
        self.ev_open_rate_hi = float(getattr(s, "AUTOTUNE_EV_OPEN_RATE_HI", 0.20))

        # Step sizes
        self.off_step = int(getattr(s, "AUTOTUNE_OFFSET_STEP_TICKS", 1))
        self.cd_step = int(getattr(s, "AUTOTUNE_COOLDOWN_STEP_MS", 2_000))
        self.rq_step = int(getattr(s, "AUTOTUNE_REQUOTE_STEP_MS", 1_000))

        # Base values (from swing config)
        self._base_cd_ms = int(getattr(s, "SWING_ENTRY_COOLDOWN_MS", 15_000))
        self._base_rq_min_ms = int(getattr(s, "SWING_REQUOTE_MIN_MS", 4_000))
        self._base_edge_mult = float(getattr(s, "SWING_MIN_EDGE_MULT", 1.2))

        # Safety autopilot (entry pauses)
        self.safety_enabled = bool(getattr(s, "SAFETY_ENABLED", True))
        self.toxic_pause_after_ms = int(getattr(s, "SAFETY_TOXIC_PAUSE_AFTER_MS", 300_000))
        self.pause_ms = int(getattr(s, "SAFETY_PAUSE_MS", 900_000))
        self.loss_limit_1h_pct = float(getattr(s, "SAFETY_LOSS_LIMIT_1H_PCT", 0.0))
        self.edge_tighten_max_mult = float(getattr(s, "SAFETY_EDGE_TIGHTEN_MAX_MULT", 1.30))

        # Live state
        self.offset_adj_ticks = 0
        self.entry_cooldown_ms = int(self._base_cd_ms)
        self.requote_min_ms = int(self._base_rq_min_ms)
        self.min_edge_mult_live = float(self._base_edge_mult)

        self.last_tune_ms = 0
        self.last_action = ""
        self.last_reason = ""

        self.trading_enabled = True
        self.pause_until_ms = 0
        self.pause_reason = ""
        self._toxic_since_ms = 0

    def _clamp(self, v: int, lo: int, hi: int) -> int:
        return max(int(lo), min(int(hi), int(v)))

    def _clampf(self, v: float, lo: float, hi: float) -> float:
        return max(float(lo), min(float(hi), float(v)))

    def _update_safety(self, now_ms: int, metrics: ExecMetrics, regime: str, equity_usdt: float) -> None:
        """Entry-only safety: pause NEW entries under prolonged toxic regime or sustained loss."""
        if not self.safety_enabled:
            self.trading_enabled = True
            self.pause_until_ms = 0
            self.pause_reason = ""
            self._toxic_since_ms = 0
            self.min_edge_mult_live = float(self._base_edge_mult)
            return

        now_ms = int(now_ms)
        regime = str(regime or "")

        # Maintain pause
        if int(self.pause_until_ms) > 0 and now_ms < int(self.pause_until_ms):
            self.trading_enabled = False
        else:
            self.trading_enabled = True
            self.pause_until_ms = 0
            self.pause_reason = ""

        # Toxic regime timer
        if regime == "TOXIC":
            if int(self._toxic_since_ms) == 0:
                self._toxic_since_ms = now_ms
            toxic_dur = now_ms - int(self._toxic_since_ms)
            if toxic_dur >= int(self.toxic_pause_after_ms) and now_ms >= int(self.pause_until_ms):
                self.trading_enabled = False
                self.pause_until_ms = now_ms + int(self.pause_ms)
                self.pause_reason = "TOXIC_PROLONGED"
        else:
            self._toxic_since_ms = 0

        # Optional loss-based pause or edge tighten
        if float(self.loss_limit_1h_pct) > 0:
            limit = -abs(float(equity_usdt)) * float(self.loss_limit_1h_pct)
            if float(metrics.pnl_1h_after_fees) < float(limit) and now_ms >= int(self.pause_until_ms):
                self.trading_enabled = False
                self.pause_until_ms = now_ms + int(self.pause_ms)
                self.pause_reason = "LOSS_1H"

        # Edge tighten (bounded) while paused due to loss (optional), otherwise reset.
        if self.pause_reason == "LOSS_1H":
            self.min_edge_mult_live = float(self._clampf(self._base_edge_mult * 1.2, self._base_edge_mult, self._base_edge_mult * self.edge_tighten_max_mult))
        else:
            self.min_edge_mult_live = float(self._base_edge_mult)

    def update(self, now_ms: int, metrics: ExecMetrics, regime: str, equity_usdt: float) -> Tuple[LiveParams, str, str]:
        """Update live execution params (bounded). Returns (live_params, action, reason)."""
        now_ms = int(now_ms)

        # Safety first
        self._update_safety(now_ms, metrics, regime=regime, equity_usdt=equity_usdt)

        # Default action is last known
        action = ""
        reason = ""

        # If paused, do not tune.
        if not bool(self.trading_enabled):
            action = "PAUSED"
            reason = str(self.pause_reason or "")
            self.last_action = action
            self.last_reason = reason
            return self.live_params(), action, reason

        if not self.enabled:
            self.last_action = ""
            self.last_reason = "DISABLED"
            return self.live_params(), "", "DISABLED"

        if int(self.last_tune_ms) > 0 and (now_ms - int(self.last_tune_ms)) < int(self.interval_ms):
            return self.live_params(), "", "THROTTLE"

        rr = float(metrics.reject_rate_5m)
        cr = float(metrics.cancel_rate_5m)
        fr = float(metrics.fill_rate_5m)
        evr = float(metrics.ev_gate_open_rate_5m)

        # One change per interval (simple & predictable)
        if rr > float(self.reject_rate_hi):
            self.offset_adj_ticks = self._clamp(int(self.offset_adj_ticks) + int(self.off_step), self.off_min, self.off_max)
            action = "OFFSET++"
            reason = f"REJECT_RATE_HIGH({rr:.2f})"
        elif cr > float(self.cancel_rate_hi):
            # Slow down if we churn cancels.
            self.entry_cooldown_ms = self._clamp(int(self.entry_cooldown_ms) + int(self.cd_step), self.cd_min, self.cd_max)
            self.requote_min_ms = self._clamp(int(self.requote_min_ms) + int(self.rq_step), self.rq_min_min, self.rq_min_max)
            action = "COOLDOWN++"
            reason = f"CANCEL_RATE_HIGH({cr:.2f})"
        elif (
            fr < float(self.fill_rate_lo)
            and evr > float(self.ev_open_rate_hi)
            and rr < float(self.reject_rate_hi) * 0.5
        ):
            # We are getting opportunities but not filling and not rejecting -> likely too far.
            self.offset_adj_ticks = self._clamp(int(self.offset_adj_ticks) - int(self.off_step), self.off_min, self.off_max)
            action = "OFFSET--"
            reason = f"FILL_RATE_LOW({fr:.2f})_EV_OPEN({evr:.2f})"
        else:
            action = ""
            reason = "NO_CHANGE"

        self.last_tune_ms = now_ms
        self.last_action = action
        self.last_reason = reason
        return self.live_params(), action, reason

    def live_params(self) -> LiveParams:
        return LiveParams(
            offset_adj_ticks=int(self.offset_adj_ticks),
            entry_cooldown_ms=int(self.entry_cooldown_ms),
            requote_min_ms=int(self.requote_min_ms),
            min_edge_mult_live=float(self.min_edge_mult_live),
            trading_enabled=bool(self.trading_enabled),
            pause_until_ms=int(self.pause_until_ms),
            pause_reason=str(self.pause_reason or ""),
        )

    def meta(self) -> Dict[str, object]:
        """Small set of meta fields for observability."""
        return {
            "autotune_enabled": bool(self.enabled),
            "autotune_last_tune_ms": int(self.last_tune_ms),
            "autotune_action": str(self.last_action or ""),
            "autotune_reason": str(self.last_reason or ""),
            "autotune_offset_adj_ticks": int(self.offset_adj_ticks),
            "autotune_entry_cooldown_ms": int(self.entry_cooldown_ms),
            "autotune_requote_min_ms": int(self.requote_min_ms),
            "min_edge_mult_live": float(self.min_edge_mult_live),
            "trading_enabled": bool(self.trading_enabled),
            "pause_until_ms": int(self.pause_until_ms),
            "pause_reason": str(self.pause_reason or ""),
        }

