from __future__ import annotations

import asyncio
import logging
import time
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Deque, Tuple, Any, Dict, List

from ..config import Settings
from ..paper.matching_engine import PaperMatchingEngine
from ..state.store import Store
from ..utils.time_ms import now_ms
from ..utils.ndjson_recorder import NDJSONRecorder
from .level4 import MicroMLPredictor, EpsilonGreedyBandit, BanditArm
from .swing_signal import SwingSignal
from .autotune import Autotuner, ExecMetrics
from .analyzer import InactivityAnalyzer

log = logging.getLogger("bot.strategy")



@dataclass
class _CalibStats:
    """Running calibration stats for predicted edge vs realized favorable move."""

    n: int = 0
    sum_pred: float = 0.0
    sum_real_fav: float = 0.0
    sum_real_dir: float = 0.0
    sum_err: float = 0.0
    sum_abs_err: float = 0.0
    sum_sq_err: float = 0.0
    hit: int = 0

    def update(self, pred_edge: float, realized_fav: float, realized_dir: float) -> None:
        try:
            pred_edge = float(pred_edge)
            realized_fav = float(realized_fav)
            realized_dir = float(realized_dir)
        except Exception:
            return
        err = float(pred_edge) - float(realized_fav)
        self.n += 1
        self.sum_pred += float(pred_edge)
        self.sum_real_fav += float(realized_fav)
        self.sum_real_dir += float(realized_dir)
        self.sum_err += float(err)
        self.sum_abs_err += abs(float(err))
        self.sum_sq_err += float(err) * float(err)
        if float(realized_dir) > 0.0:
            self.hit += 1

    def summary(self) -> Dict[str, float]:
        n = int(self.n)
        if n <= 0:
            return {
                "n": 0,
                "pred_mean": 0.0,
                "real_fav_mean": 0.0,
                "real_dir_mean": 0.0,
                "bias": 0.0,
                "mae": 0.0,
                "rmse": 0.0,
                "hit_rate": 0.0,
            }
        pred_mean = float(self.sum_pred) / float(n)
        real_fav_mean = float(self.sum_real_fav) / float(n)
        real_dir_mean = float(self.sum_real_dir) / float(n)
        bias = float(self.sum_err) / float(n)
        mae = float(self.sum_abs_err) / float(n)
        rmse = math.sqrt(float(self.sum_sq_err) / float(n)) if float(self.sum_sq_err) > 0 else 0.0
        hit_rate = float(self.hit) / float(n)
        return {
            "n": float(n),
            "pred_mean": float(pred_mean),
            "real_fav_mean": float(real_fav_mean),
            "real_dir_mean": float(real_dir_mean),
            "bias": float(bias),
            "mae": float(mae),
            "rmse": float(rmse),
            "hit_rate": float(hit_rate),
        }


@dataclass
class _CalibBin:
    lo: int = 0
    hi: int = 0
    n: int = 0
    sum_pred: float = 0.0
    sum_real_fav: float = 0.0
    hit: int = 0

    def update(self, pred_edge: float, realized_fav: float, realized_dir: float) -> None:
        try:
            pred_edge = float(pred_edge)
            realized_fav = float(realized_fav)
            realized_dir = float(realized_dir)
        except Exception:
            return
        self.n += 1
        self.sum_pred += float(pred_edge)
        self.sum_real_fav += float(realized_fav)
        if float(realized_dir) > 0.0:
            self.hit += 1

    def to_dict(self) -> Dict[str, Any]:
        n = int(self.n)
        if n <= 0:
            return {
                "lo": int(self.lo),
                "hi": int(self.hi),
                "n": 0,
                "pred_mean": 0.0,
                "real_fav_mean": 0.0,
                "hit_rate": 0.0,
            }
        return {
            "lo": int(self.lo),
            "hi": int(self.hi),
            "n": int(n),
            "pred_mean": float(self.sum_pred) / float(n),
            "real_fav_mean": float(self.sum_real_fav) / float(n),
            "hit_rate": float(self.hit) / float(n),
        }

@dataclass
class StrategyState:
    entry_ts_ms: int = 0
    entry_price: float = 0.0
    side: str = ""  # LONG/SHORT
    cooldown_until_ms: int = 0

    # Swing mode state
    swing_entry_cooldown_until_ms: int = 0
    swing_last_requote_ms: int = 0
    swing_exit_started_ms: int = 0
    swing_exit_last_requote_ms: int = 0
    swing_exit_requotes: int = 0
    swing_last_entry_attempt_ms: int = 0
    swing_exit_reason: str = ""
    swing_gate_off_since_ms: int = 0

    # Phase 3.1b: trailing profit state (fee-cover activation)
    swing_trail_active: bool = False
    swing_trail_peak_ticks: int = 0
    swing_trail_retrace_ticks: int = 0
    swing_trail_activate_ticks: int = 0
    swing_trail_mode: str = ""
    swing_trail_be_floor_ticks: int = 0
    swing_trail_letrun_on: bool = False

    # Level-1 throttles (derived from CLOSED trade outcomes)
    trades_seen: int = 0
    loss_streak: int = 0
    last_quote_ts_ms: int = 0

    # Level-2 regime tracking (rolling window)
    regime_mid_hist: Deque[Tuple[int, float]] = field(default_factory=lambda: deque(maxlen=400))
    regime_imb_hist: Deque[Tuple[int, float]] = field(default_factory=lambda: deque(maxlen=400))
    # v2: baseline distributions for toxic score
    regime_spread_hist: Deque[Tuple[int, float]] = field(default_factory=lambda: deque(maxlen=400))
    regime_vol_ticks_hist: Deque[Tuple[int, float]] = field(default_factory=lambda: deque(maxlen=400))
    regime_churn_ts: Deque[int] = field(default_factory=lambda: deque(maxlen=800))
    regime_last_bb_tick: int = 0
    regime_last_ba_tick: int = 0
    regime_last_log_ms: int = 0

    # Per-side quote pacing (Level-2 feedback loop)
    last_quote_buy_ts_ms: int = 0
    last_quote_sell_ts_ms: int = 0

    # Level-3 market making state
    last_pos_qty: float = 0.0
    pos_open_ts_ms: int = 0
    last_regime_class: str = ""
    last_mm_bid_inside: int = 0
    last_mm_ask_inside: int = 0
    last_requote_buy_ts_ms: int = 0
    last_requote_sell_ts_ms: int = 0

    # Level-4 meta / ML probes
    last_meta_ts_ms: int = 0
    last_ml_probe_ts_ms: int = 0

    # --- Phase 2: sizing + side performance ---
    side_perf_hist: Deque[Tuple[int, str, float]] = field(default_factory=lambda: deque(maxlen=2000))
    equity_peak: float = 0.0
    size_mult: float = 1.0
    size_last_inc_ms: int = 0

    # Phase 0: observability (why we are idle / not trading)
    status_last_log_ms: int = 0
    last_no_trade_reason: str = ""

    # Phase 1: liveness (strategy_meta schema stability)
    last_step_ts_ms: int = 0
    loop_dt_ms: int = 0

    # --- Phase 1.5: rolling execution metrics (entry-only) ---
    entry_attempt_ts: Deque[int] = field(default_factory=lambda: deque(maxlen=5000))
    entry_fill_ts: Deque[int] = field(default_factory=lambda: deque(maxlen=5000))
    loop_ts_5m: Deque[int] = field(default_factory=lambda: deque(maxlen=20000))
    ev_gate_open_ts: Deque[int] = field(default_factory=lambda: deque(maxlen=20000))
    pending_entry_ts_ms: int = 0
    time_to_fill_ms: Deque[int] = field(default_factory=lambda: deque(maxlen=400))
    closed_seen: int = 0
    pnl_events_1h: Deque[Tuple[int, float]] = field(default_factory=lambda: deque(maxlen=5000))

    # Safety pause housekeeping (avoid cancel-spam while paused)
    pause_last_cancel_ms: int = 0

    # --- Phase 3.3: prediction audit / calibration ---
    pred_audit_last_sample_ms: int = 0
    pred_audit_60: Deque[Tuple[int, float, float, int, float]] = field(default_factory=lambda: deque(maxlen=6000))
    pred_audit_120: Deque[Tuple[int, float, float, int, float]] = field(default_factory=lambda: deque(maxlen=9000))
    pred_audit_300: Deque[Tuple[int, float, float, int, float]] = field(default_factory=lambda: deque(maxlen=18000))
    calib_60: _CalibStats = field(default_factory=_CalibStats)
    calib_120: _CalibStats = field(default_factory=_CalibStats)
    calib_300: _CalibStats = field(default_factory=_CalibStats)


class MicroScalpStrategy:
    """Maker-only micro-scalp.

    This is intentionally conservative (safe-by-default), but the defaults are
    tuned to be *more active on testnet* where aggTrade flow can be sparse.

    Stability rules:
      - never place a new order if there is any open order (NEW/PARTIALLY_FILLED)
      - cancel stale open orders so we don't get stuck
    """

    def __init__(
        self,
        store: Store,
        engine: PaperMatchingEngine,
        settings: Settings,
        *,
        recorder: Optional[NDJSONRecorder] = None,
    ) -> None:
        self.store = store
        self.engine = engine
        self.s = settings
        self.state = StrategyState()
        self.recorder = recorder
        self._task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()

        # Phase 2.1: NDJSON recording downsampling
        self._recorder_frame_ms = int(getattr(self.s, "RECORDER_FRAME_MS", 250) or 250)
        self._recorder_last_frame_ms = 0


        # Phase 3.3: online prediction audit / calibration
        self._calib_enabled = bool(getattr(self.s, "CALIB_ENABLED", True))
        self._calib_sample_ms = int(getattr(self.s, "CALIB_SAMPLE_MS", self._recorder_frame_ms) or self._recorder_frame_ms)
        # bins for 120s calibration (pred_edge_ticks_est -> realized favorable move)
        self._calib_bin_edges: List[int] = [0, 8, 12, 16, 24, 32, 48, 64, 96, 9999]
        self._calib_bins_120: List[_CalibBin] = [
            _CalibBin(lo=int(self._calib_bin_edges[i]), hi=int(self._calib_bin_edges[i + 1]))
            for i in range(len(self._calib_bin_edges) - 1)
        ]

        # runtime-tunable knobs (config defaults tuned for testnet activity)
        self._order_stale_ms = int(getattr(self.s, "ORDER_STALE_MS", 12000))
        self._max_vol = float(getattr(self.s, "STRATEGY_MAX_VOL", 0.0008))
        self._entry_score = float(getattr(self.s, "STRATEGY_ENTRY_SCORE", 0.15))
        self._peg_inside = int(getattr(self.s, "STRATEGY_PEG_INSIDE_TICKS", 1))
        self._cooldown_after_stop_ms = int(getattr(self.s, "STRATEGY_COOLDOWN_AFTER_STOP_MS", 10_000))

        # Strategy mode (default: swing_1m)
        self._strategy_mode = str(getattr(self.s, "STRATEGY_MODE", "swing_1m") or "swing_1m")

        # Phase 0: low-noise status logs (why we are idle)
        self._status_log_every_ms = int(getattr(self.s, "STRATEGY_STATUS_LOG_MS", 2500) or 2500)

        # Swing signal + params
        self._swing = SwingSignal(
            max_hist_ms=int(getattr(self.s, "SWING_MAX_HOLD_MS", 300000)) + 120_000,
            min_dwell_ms=10_000,
            flip_stronger_mult=1.5,
        )

        self._swing_min_edge_mult = float(getattr(self.s, "SWING_MIN_EDGE_MULT", 1.2))
        self._swing_entry_base_off = int(getattr(self.s, "SWING_ENTRY_BASE_OFFSET_TICKS", 8))
        self._swing_entry_off_fees = float(getattr(self.s, "SWING_ENTRY_OFFSET_MULT_FEES", 0.7))
        self._swing_entry_off_vol = float(getattr(self.s, "SWING_ENTRY_OFFSET_MULT_VOL", 0.8))
        self._swing_entry_cd_ms = int(getattr(self.s, "SWING_ENTRY_COOLDOWN_MS", 15000))

        self._swing_min_order_life_ms = int(getattr(self.s, "SWING_MIN_ORDER_LIFE_MS", 2000))
        self._swing_rq_min_ms = int(getattr(self.s, "SWING_REQUOTE_MIN_MS", 4000))
        self._swing_rq_drift_ticks = int(getattr(self.s, "SWING_REQUOTE_DRIFT_TICKS", 8))
        self._swing_gate_hold_ms = int(getattr(self.s, "SWING_GATE_HOLD_MS", 2000))

        # Phase 3.4: exit execution discipline (maker-only, throttled repricing)
        self._swing_exit_patience_ms = max(1500, int(getattr(self.s, "SWING_EXIT_PATIENCE_MS", 2500)))
        self._swing_exit_requote_min_ms = max(1500, int(getattr(self.s, "SWING_EXIT_REQUOTE_MIN_MS", 2500)))
        self._swing_exit_rq_drift_ticks = max(1, int(getattr(self.s, "SWING_EXIT_REQUOTE_DRIFT_TICKS", 1)))

        self._swing_tp_mult_fees = float(getattr(self.s, "SWING_TP_MULT_FEES", 2.0))
        self._swing_tp_min_ticks = int(getattr(self.s, "SWING_TP_TICKS_MIN", 16))
        self._swing_sl_mult_fees = float(getattr(self.s, "SWING_SL_MULT_FEES", 1.5))
        self._swing_sl_min_ticks = int(getattr(self.s, "SWING_SL_TICKS_MIN", 12))
        self._swing_max_hold_ms = int(getattr(self.s, "SWING_MAX_HOLD_MS", 300000))

        self._swing_risk_pct = float(getattr(self.s, "SWING_RISK_PCT", 0.005))
        self._swing_emergency = bool(getattr(self.s, "SWING_EMERGENCY_EXIT", False))

        # Phase 3.1b: fee-cover activation + trailing exit (let winners run)
        self._swing_trail_enable = bool(getattr(self.s, "SWING_TRAIL_ENABLE", True))
        self._swing_trail_activate_mult_fees = float(getattr(self.s, "SWING_TRAIL_ACTIVATE_MULT_FEES", 2.0))
        self._swing_trail_min_activate_ticks = int(getattr(self.s, "SWING_TRAIL_MIN_ACTIVATE_TICKS", 8))
        self._swing_trail_dd_pct = float(getattr(self.s, "SWING_TRAIL_DD_PCT", 0.10))
        self._swing_trail_atr_k = float(getattr(self.s, "SWING_TRAIL_ATR_K", 0.9))
        self._swing_trail_min_retrace_ticks = int(getattr(self.s, "SWING_TRAIL_MIN_RETRACE_TICKS", 8))
        self._swing_trail_min_peak_ticks = int(getattr(self.s, "SWING_TRAIL_MIN_PEAK_TICKS", 0))

        # Phase 3.3c: protect then let it run
        self._swing_trail_be_mult = float(getattr(self.s, "SWING_TRAIL_BE_MULT", 1.3))
        self._swing_trail_be_buffer_ticks = int(getattr(self.s, "SWING_TRAIL_BE_BUFFER_TICKS", 3))
        self._swing_trail_letrun_after_ms = int(getattr(self.s, "SWING_TRAIL_LETRUN_AFTER_MS", 45_000))
        self._swing_trail_letrun_k_mult = float(getattr(self.s, "SWING_TRAIL_LETRUN_K_MULT", 1.25))

        # Phase 3.3d: anti-loser exits (early adverse selection + time-decay) — bounded, maker-only.
        self._swing_early_as_enable = bool(getattr(self.s, "SWING_EARLY_AS_ENABLE", True))
        self._swing_early_as_window_ms = int(getattr(self.s, "SWING_EARLY_AS_WINDOW_MS", 25_000))
        self._swing_early_as_loss_ticks = int(getattr(self.s, "SWING_EARLY_AS_LOSS_TICKS", 12))
        self._swing_early_as_toxic_score = float(getattr(self.s, "SWING_EARLY_AS_TOXIC_SCORE", 1.2))
        self._swing_early_as_churn_ps = float(getattr(self.s, "SWING_EARLY_AS_CHURN_PER_S", 10.0))
        self._swing_early_as_inside_add = int(getattr(self.s, "SWING_EARLY_AS_INSIDE_ADD_TICKS", 1))

        self._swing_time_decay_enable = bool(getattr(self.s, "SWING_TIME_DECAY_ENABLE", True))
        self._swing_time_decay_start_ms = int(getattr(self.s, "SWING_TIME_DECAY_START_MS", 60_000))
        self._swing_time_decay_step_ms = int(getattr(self.s, "SWING_TIME_DECAY_STEP_MS", 30_000))
        self._swing_time_decay_max_level = int(getattr(self.s, "SWING_TIME_DECAY_MAX_LEVEL", 3))
        self._swing_time_decay_min_loss_ticks = int(getattr(self.s, "SWING_TIME_DECAY_MIN_LOSS_TICKS", 6))
        self._swing_time_decay_inside_add_per_level = int(getattr(self.s, "SWING_TIME_DECAY_INSIDE_ADD_PER_LEVEL", 1))
        self._swing_time_decay_max_inside_add = int(getattr(self.s, "SWING_TIME_DECAY_MAX_INSIDE_ADD_TICKS", 4))

        # Phase 1.5: bounded execution autotuner + inactivity analyzer (diagnostics-only)
        self._autotune = Autotuner(self.s)
        self._analyzer = InactivityAnalyzer()

        # --- Level 1 throttles (dynamic cooldown + soft-stop) ---
        self._dyn_cd_streak = int(getattr(self.s, "STRATEGY_DYN_COOLDOWN_STREAK", 3))
        self._dyn_cd_min_ms = int(getattr(self.s, "STRATEGY_DYN_COOLDOWN_MIN_MS", 10_000))
        self._dyn_cd_max_ms = int(getattr(self.s, "STRATEGY_DYN_COOLDOWN_MAX_MS", 60_000))

        self._softstop_streak = int(getattr(self.s, "STRATEGY_SOFTSTOP_STREAK", 2))
        self._softstop_qty_mult = float(getattr(self.s, "STRATEGY_SOFTSTOP_QTY_MULT", 0.33))
        self._softstop_quote_interval_ms = int(getattr(self.s, "STRATEGY_SOFTSTOP_QUOTE_INTERVAL_MS", 400))
        self._softstop_entry_mult = float(getattr(self.s, "STRATEGY_SOFTSTOP_ENTRY_SCORE_MULT", 1.25))

        # --- Level 2: regime filter + fill-quality feedback (still simple) ---
        self._regime_window_ms = int(getattr(self.s, "STRATEGY_REGIME_WINDOW_MS", 5000))
        self._regime_max_churn_ps = float(getattr(self.s, "STRATEGY_REGIME_MAX_CHURN_PER_S", 8.0))
        self._regime_max_imb_std = float(getattr(self.s, "STRATEGY_REGIME_MAX_IMB_STD", 0.35))
        self._regime_min_spread_to_vol = float(getattr(self.s, "STRATEGY_REGIME_MIN_SPREAD_TO_VOL", 0.6))
        self._regime_trend_ticks = int(getattr(self.s, "STRATEGY_REGIME_TREND_TICKS", 4))

        self._fq_horizon_ms = int(getattr(self.s, "STRATEGY_FQ_HORIZON_MS", 500))
        self._fq_min_n = int(getattr(self.s, "STRATEGY_FQ_MIN_N", 30))
        self._fq_edge_scale = float(getattr(self.s, "STRATEGY_FQ_EDGE_SCALE_TICKS", 0.5))
        self._fq_neg_entry_mult_max = float(getattr(self.s, "STRATEGY_FQ_NEG_ENTRY_MULT_MAX", 2.0))
        self._fq_neg_quote_int_max_ms = int(getattr(self.s, "STRATEGY_FQ_NEG_QUOTE_INTERVAL_MAX_MS", 600))

        # --- Level 3: microstructure upgrades ---
        self._mm_enabled = bool(getattr(self.s, "STRATEGY_MM_ENABLED", True))
        self._mm_base_inside = int(getattr(self.s, "STRATEGY_MM_BASE_INSIDE_TICKS", self._peg_inside))
        self._mm_max_inside = int(getattr(self.s, "STRATEGY_MM_MAX_INSIDE_TICKS", 3))
        self._mm_signal_skew_ticks = int(getattr(self.s, "STRATEGY_MM_SIGNAL_SKEW_TICKS", 2))
        self._mm_inv_skew_ticks = int(getattr(self.s, "STRATEGY_MM_INV_SKEW_TICKS", 2))
        self._mm_inv_max_pos_qty = float(getattr(self.s, "STRATEGY_MM_INV_MAX_POS_QTY", float(self.s.ORDER_QTY) * 3.0))
        self._mm_inv_add_qty_mult = float(getattr(self.s, "STRATEGY_MM_INV_ADD_QTY_MULT", 0.5))
        self._mm_fq_widen_ticks = int(getattr(self.s, "STRATEGY_MM_FQ_WIDEN_TICKS", 2))
        self._mm_requote_price_ticks = int(getattr(self.s, "STRATEGY_MM_REQUOTE_PRICE_TICKS", 2))
        self._mm_requote_min_ms = int(getattr(self.s, "STRATEGY_MM_REQUOTE_MIN_INTERVAL_MS", 400))

        # --- Phase 1: fee-aware quoting / gating ---
        self._fee_buffer_ticks = int(getattr(self.s, "STRATEGY_FEE_BUFFER_TICKS", 1))
        self._fee_add_extra_ticks = int(getattr(self.s, "STRATEGY_FEE_ADD_EXTRA_TICKS", 1))

        # Anti-flap: do not cancel/replace very fresh orders on small drifts
        self._mm_min_order_life_ms = int(getattr(self.s, "STRATEGY_MM_MIN_ORDER_LIFE_MS", 400))

        # --- Phase 2: dynamic sizing + side performance control ---
        # Notional cap for ENTRY/ADD (reduce-side exits are always allowed to close full position).
        self._max_notional_pct = float(getattr(self.s, "STRATEGY_MAX_NOTIONAL_PCT", 0.10))


        # Base order notional sizing (if >0, overrides ORDER_QTY as the starting point)
        self._base_notional_pct = float(getattr(self.s, "STRATEGY_BASE_NOTIONAL_PCT", 0.0))

        # Relax strict fee gate when fees dominate spread (testnet usability)
        self._relax_fee_gate = bool(getattr(self.s, "STRATEGY_RELAX_FEE_GATE", True))
        self._relax_gate_spread_mult = float(getattr(self.s, "STRATEGY_RELAX_GATE_SPREAD_MULT", 0.50))

        # Sticky quotes reduce churn by avoiding constant cancel/replace chasing
        self._mm_sticky_quotes = bool(getattr(self.s, "STRATEGY_MM_STICKY_QUOTES", True))
        self._size_min_mult = float(getattr(self.s, "STRATEGY_SIZE_MIN_MULT", 0.25))
        self._size_max_mult = float(getattr(self.s, "STRATEGY_SIZE_MAX_MULT", 2.50))
        self._size_edge_exp = float(getattr(self.s, "STRATEGY_SIZE_EDGE_EXP", 1.40))
        # edge/min_inside at which we reach SIZE_MAX_MULT
        self._size_edge_full_at = float(getattr(self.s, "STRATEGY_SIZE_EDGE_FULL_AT", 1.00))
        self._size_regime_toxic_mult = float(getattr(self.s, "STRATEGY_SIZE_REGIME_TOXIC_MULT", 0.50))
        self._size_regime_trend_mult = float(getattr(self.s, "STRATEGY_SIZE_REGIME_TREND_MULT", 1.10))
        self._size_dd_cutoff = float(getattr(self.s, "STRATEGY_SIZE_DD_CUTOFF", 0.01))
        self._size_dd_min_mult = float(getattr(self.s, "STRATEGY_SIZE_DD_MIN_MULT", 0.35))
        self._size_grow_cooldown_ms = int(getattr(self.s, "STRATEGY_SIZE_GROW_COOLDOWN_MS", 5000))
        self._size_grow_step = float(getattr(self.s, "STRATEGY_SIZE_GROW_STEP", 0.15))
        self._size_max_qty_mult = float(getattr(self.s, "STRATEGY_SIZE_MAX_QTY_MULT", 3.00))

        self._side_perf_window_ms = int(getattr(self.s, "STRATEGY_SIDE_PERF_WINDOW_MS", 600_000))
        self._side_perf_min_trades = int(getattr(self.s, "STRATEGY_SIDE_PERF_MIN_TRADES", 5))
        self._side_perf_target_usdt = float(getattr(self.s, "STRATEGY_SIDE_PERF_TARGET_USDT", 0.02))
        self._side_perf_min_mult = float(getattr(self.s, "STRATEGY_SIDE_PERF_MIN_MULT", 0.30))
        self._side_perf_max_mult = float(getattr(self.s, "STRATEGY_SIDE_PERF_MAX_MULT", 1.20))

        # Phase 2c: extend side control with FillQ + per-side gate/inside adjustments + gate-hold.
        self._side_use_fillq = bool(getattr(self.s, "STRATEGY_SIDE_USE_FILLQ", True))
        self._side_fq_weight = float(getattr(self.s, "STRATEGY_SIDE_FQ_WEIGHT", 0.35))
        self._side_gate_max_mult = float(getattr(self.s, "STRATEGY_SIDE_GATE_MAX_MULT", 1.75))
        self._side_inside_adj_max = int(getattr(self.s, "STRATEGY_SIDE_INSIDE_ADJ_MAX_TICKS", 2))
        self._gate_hold_ms = int(getattr(self.s, "STRATEGY_GATE_HOLD_MS", 2000))


        # --- Level 4: micro-ML + bandit (no LLM) ---
        self._ml_enabled = bool(getattr(self.s, "STRATEGY_ML_ENABLED", True))
        self._ml_prefer_h = int(getattr(self.s, "STRATEGY_ML_PREFER_H_MS", 500))
        self._ml_warmup_n = int(getattr(self.s, "STRATEGY_ML_WARMUP_N", 200))
        self._ml_min_edge = float(getattr(self.s, "STRATEGY_ML_MIN_EDGE", 0.06))
        self._ml_one_side_edge = float(getattr(self.s, "STRATEGY_ML_ONE_SIDED_EDGE", 0.12))
        self._ml_skew_ticks = int(getattr(self.s, "STRATEGY_ML_SKEW_TICKS", 2))
        self._ml_probe_interval_ms = int(getattr(self.s, "STRATEGY_ML_PROBE_INTERVAL_MS", 120))
        self._ml_lr = float(getattr(self.s, "STRATEGY_ML_LR", 0.05))
        self._ml_l2 = float(getattr(self.s, "STRATEGY_ML_L2", 1e-4))

        self._ml = MicroMLPredictor(horizons_ms=(200, 500, 1000), feature_dim=8, lr=self._ml_lr, l2=self._ml_l2) if self._ml_enabled else None

        self._bandit_enabled = bool(getattr(self.s, "STRATEGY_BANDIT_ENABLED", True))
        self._bandit_switch_ms = int(getattr(self.s, "STRATEGY_BANDIT_SWITCH_MS", 30000))
        self._bandit_eps = float(getattr(self.s, "STRATEGY_BANDIT_EPSILON", 0.15))
        self._bandit_eps_min = float(getattr(self.s, "STRATEGY_BANDIT_EPS_MIN", 0.03))
        self._bandit_eps_decay = float(getattr(self.s, "STRATEGY_BANDIT_EPS_DECAY", 0.995))
        self._bandit_alpha = float(getattr(self.s, "STRATEGY_BANDIT_EWMA_ALPHA", 0.25))

        self._bandit = None
        self._bandit_last_net = 0.0
        self._bandit_last_switch_ms = 0
        if self._bandit_enabled:
            base_inside0 = int(self._mm_base_inside)
            base_sig0 = int(self._mm_signal_skew_ticks)
            # 5 conservative arms (avoid cancel-storm)
            arms = [
                BanditArm(name="base", base_inside_ticks=base_inside0, signal_skew_ticks=base_sig0, requote_price_ticks=int(self._mm_requote_price_ticks), requote_min_ms=int(self._mm_requote_min_ms), qty_mult=1.0, min_edge_mult=1.0),
                BanditArm(name="tight", base_inside_ticks=max(0, base_inside0 - 1), signal_skew_ticks=base_sig0, requote_price_ticks=max(2, int(self._mm_requote_price_ticks) - 1), requote_min_ms=max(600, int(self._mm_requote_min_ms)), qty_mult=1.0, min_edge_mult=1.15),
                BanditArm(name="wide", base_inside_ticks=min(int(self._mm_max_inside), base_inside0 + 1), signal_skew_ticks=base_sig0, requote_price_ticks=int(self._mm_requote_price_ticks) + 1, requote_min_ms=int(self._mm_requote_min_ms) + 200, qty_mult=1.0, min_edge_mult=0.90),
                BanditArm(name="defensive", base_inside_ticks=min(int(self._mm_max_inside), base_inside0 + 2), signal_skew_ticks=max(1, base_sig0 - 1), requote_price_ticks=int(self._mm_requote_price_ticks) + 1, requote_min_ms=int(self._mm_requote_min_ms) + 700, qty_mult=0.7, min_edge_mult=1.35),
                BanditArm(name="aggr", base_inside_ticks=base_inside0, signal_skew_ticks=base_sig0 + 1, requote_price_ticks=max(2, int(self._mm_requote_price_ticks) - 1), requote_min_ms=max(500, int(self._mm_requote_min_ms) - 200), qty_mult=1.15, min_edge_mult=0.90),
            ]
            self._bandit = EpsilonGreedyBandit(
                arms=arms,
                epsilon=self._bandit_eps,
                epsilon_min=self._bandit_eps_min,
                epsilon_decay=self._bandit_eps_decay,
                ewma_alpha=self._bandit_alpha,
                switch_ms=self._bandit_switch_ms,
            )

    def _clamp_int(self, x: int, lo: int, hi: int) -> int:
        return int(max(int(lo), min(int(hi), int(x))))

    def _clamp_float(self, x: float, lo: float, hi: float) -> float:
        return float(max(float(lo), min(float(hi), float(x))))

    def _maker_price_inside(self, side: str, bb: float, ba: float, tick: float, inside_ticks: int) -> float:
        """Maker-only price with a configurable inside-ticks (0..N)."""
        eps = max(float(tick) * 0.25, 1e-9)
        inside = max(0, int(inside_ticks))
        if side == "BUY":
            base = float(bb)
            if inside <= 0:
                return base
            candidate = float(bb) + inside * float(tick)
            return candidate if candidate < float(ba) - eps else base
        # SELL
        base = float(ba)
        if inside <= 0:
            return base
        candidate = float(ba) - inside * float(tick)
        return candidate if candidate > float(bb) + eps else base

    def _open_order_by_side(self, snap: dict) -> dict:
        out = {"BUY": None, "SELL": None}
        for o in (snap.get("orders", []) or []):
            if o.get("status") not in ("NEW", "PARTIALLY_FILLED"):
                continue
            side = str(o.get("side", ""))
            if side not in out:
                continue
            # keep the most recent for that side
            if out[side] is None or int(o.get("ts_ms", 0) or 0) > int(out[side].get("ts_ms", 0) or 0):
                out[side] = o
        return out

    def start(self) -> None:
        if self._task and not self._task.done():
            return
        self._stop.clear()
        self._task = asyncio.create_task(self._run(), name="strategy_loop")

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            await asyncio.wait([self._task], timeout=5)

    def _now_ms(self) -> int:
        return int(now_ms())

    def _maybe_status_log(
        self,
        now_ms: int,
        *,
        mode: str,
        symbol: str,
        bot_running: bool,
        kill_switch: bool,
        reason: str,
        in_pos: bool,
        pos_qty: float,
        avg_px: float,
        open_orders: int,
        swing_dir: int,
        swing_strength: float,
        swing_edge_ticks: float,
        swing_score_ticks: float = 0.0,
        trade_pressure: float = 0.0,
        fee_ticks: int,
        min_profit_ticks: int,
        activate_ticks: int = 0,
        regime: str,
        cooldown_left_ms: int,
        trail_active: bool = False,
        trail_peak_ticks: int = 0,
        pos_delta_ticks: int = 0,
    ) -> None:
        """Low-noise, periodic state log so you can always see *why* we are idle."""
        if int(now_ms) - int(self.state.status_last_log_ms) < int(self._status_log_every_ms):
            return
        self.state.status_last_log_ms = int(now_ms)
        self.state.last_no_trade_reason = str(reason or "")

        rec_lines = rec_drop = rec_q = 0
        if self.recorder is not None:
            try:
                rd = self.recorder.meta_dict()
                rec_lines = int(rd.get("recorder_lines") or 0)
                rec_drop = int(rd.get("recorder_dropped") or 0)
                rec_q = int(rd.get("recorder_queue_len") or 0)
            except Exception:
                pass
        log.info(
            "status mode=%s symbol=%s bot_running=%s kill=%s in_pos=%s pos=%.6f avg=%.2f open=%d "
            "dir=%d str=%.2f edge=%.1f score=%.1f press=%.2f fee=%d minp=%d regime=%s cd_ms=%d "
            "act=%d trail=%s peak=%d dlt=%d rec_lines=%d rec_drop=%d rec_q=%d reason=%s",
            str(mode),
            str(symbol),
            bool(bot_running),
            bool(kill_switch),
            bool(in_pos),
            float(pos_qty),
            float(avg_px),
            int(open_orders),
            int(swing_dir),
            float(swing_strength),
            float(swing_edge_ticks),
            float(swing_score_ticks),
            float(trade_pressure),
            int(fee_ticks),
            int(min_profit_ticks),
            str(regime),
            int(cooldown_left_ms),
            int(activate_ticks),
            bool(trail_active),
            int(trail_peak_ticks),
            int(pos_delta_ticks),
            int(rec_lines),
            int(rec_drop),
            int(rec_q),
            str(reason or ""),
        )

    def _maybe_record_frame(
        self,
        now_ms: int,
        snap: dict,
        *,
        pred: Optional[Any],
        meta: Dict[str, Any],
        trade_pressure: float = 0.0,
        churn_ps: float = 0.0,
    ) -> None:
        """Phase 2.1 NDJSON recorder: downsampled frames for offline reproduction.

        This is intentionally *light*: one frame every RECORDER_FRAME_MS.
        """
        if self.recorder is None or not bool(getattr(self.recorder, "enabled", False)):
            return
        if int(now_ms) - int(self._recorder_last_frame_ms) < int(self._recorder_frame_ms):
            return
        self._recorder_last_frame_ms = int(now_ms)

        m = (snap.get("market") or {}) if isinstance(snap, dict) else {}
        trades = (snap.get("trades") or []) if isinstance(snap, dict) else []
        last_trade = None
        try:
            if isinstance(trades, list) and trades:
                t = trades[0]
                last_trade = {
                    "ts_ms": int(t.get("ts_ms") or 0),
                    "price": float(t.get("price") or 0.0),
                    "qty": float(t.get("qty") or 0.0),
                    "side": str(t.get("side") or ""),
                }
        except Exception:
            last_trade = None

        book = {
            "bid": float(m.get("best_bid") or 0.0),
            "bid_qty": float(m.get("best_bid_qty") or 0.0),
            "ask": float(m.get("best_ask") or 0.0),
            "ask_qty": float(m.get("best_ask_qty") or 0.0),
            "mid": float(m.get("mid") or 0.0),
            "spread_ticks": float(m.get("spread_ticks") or 0.0),
            "tick_size": float(m.get("tick_size") or 0.0),
        }

        swing = None
        if pred is not None:
            try:
                swing = {
                    "dir": int(getattr(pred, "dir", 0) or 0),
                    "strength": float(getattr(pred, "strength", 0.0) or 0.0),
                    "edge_ticks_est": float(getattr(pred, "edge_ticks_est", 0.0) or 0.0),
                    "score_ticks": float(getattr(pred, "score_ticks", 0.0) or 0.0),
                    "atr_ticks_120": float(getattr(pred, "atr_ticks_120", getattr(pred, "atr_ticks", 0.0)) or 0.0),
                    "slope_norm": float(getattr(pred, "slope_norm", 0.0) or 0.0),
                    "imb_ema": float(getattr(pred, "imbalance_ema", 0.0) or 0.0),
                    "imb_std": float(getattr(pred, "imb_std", 0.0) or 0.0),
                    "mark_last_div_ticks": float(getattr(pred, "mark_last_div_ticks", getattr(pred, "mark_divergence_ticks", 0.0)) or 0.0),
                    "atr_ticks": float(getattr(pred, "atr_ticks", 0.0) or 0.0),
                    "slope_ticks": float(getattr(pred, "slope_ticks", 0.0) or 0.0),
                    "imbalance_ema": float(getattr(pred, "imbalance_ema", 0.0) or 0.0),
                    "pressure_ema": float(getattr(pred, "pressure_ema", 0.0) or 0.0),
                    "churn_ema": float(getattr(pred, "churn_ema", 0.0) or 0.0),
                    "spread_ticks": float(getattr(pred, "spread_ticks", 0.0) or 0.0),
                    "mark_divergence_ticks": float(getattr(pred, "mark_divergence_ticks", 0.0) or 0.0),
                }
            except Exception:
                swing = None

        # decisions/gates (keep small but sufficient for replay)
        acct = (snap.get("account") or {}) if isinstance(snap, dict) else {}
        try:
            pos_qty = float(acct.get("position_qty") or 0.0)
            avg_px = float(acct.get("avg_entry_price") or 0.0)
            in_pos = bool(abs(float(pos_qty)) > 1e-12)
        except Exception:
            pos_qty = 0.0
            avg_px = 0.0
            in_pos = False
        decision = {
            "no_trade_reason": str(meta.get("no_trade_reason") or ""),
            "gate_state": str(meta.get("gate_state") or ""),
            "desired_side": str(meta.get("desired_side") or ""),
            "entry_price": float(meta.get("entry_price") or 0.0),
            "entry_offset_ticks": int(meta.get("entry_offset_ticks") or 0),
            "qty_used": float(meta.get("qty_used") or 0.0),
            "in_pos": bool(in_pos),
            "pos_qty": float(pos_qty),
            "avg_px": float(avg_px),
            "hold_ms": int(meta.get("hold_ms") or 0),
            "max_hold_ms": int(meta.get("max_hold_ms") or 0),
            # Phase 3.1b: trailing profit observability for replay
            "trail_enabled": bool(meta.get("trail_enabled") or False),
            "trail_active": bool(meta.get("trail_active") or False),
            "trail_activate_ticks": int(meta.get("trail_activate_ticks") or 0),
            "trail_peak_ticks": int(meta.get("trail_peak_ticks") or 0),
            "trail_retrace_ticks": int(meta.get("trail_retrace_ticks") or 0),
            "trail_trigger_level_ticks": int(meta.get("trail_trigger_level_ticks") or 0),
            "trail_dist_ticks": int(meta.get("trail_dist_ticks") or meta.get("trail_retrace_ticks") or 0),
            "trail_trigger_ticks": int(meta.get("trail_trigger_ticks") or meta.get("trail_trigger_level_ticks") or 0),
            "atr_ticks": float(meta.get("trail_atr_ticks") or meta.get("atr_ticks_120") or 0.0),
            "dd_pct": float(meta.get("trail_dd_pct") or 0.0),
            "k": float(meta.get("trail_k") or 0.0),
            # Phase 3.3c: protect/let-run state
            "trail_mode": str(meta.get("trail_mode") or ""),
            "be_floor_ticks": int(meta.get("be_floor_ticks") or 0),
            "letrun_on": bool(meta.get("letrun_on") or False),
            "letrun_after_ms": int(meta.get("letrun_after_ms") or 0),

            # Phase 3.3d: anti-loser observability for replay
            "early_as_triggered": bool(meta.get("early_as_triggered") or False),
            "early_as_loss_ticks": int(meta.get("early_as_loss_ticks") or 0),
            "early_as_window_ms": int(meta.get("early_as_window_ms") or 0),
            "time_decay_active": bool(meta.get("time_decay_active") or False),
            "time_decay_level": int(meta.get("time_decay_level") or 0),

            # Phase 3.4: exit execution discipline for offline replay
            "exit_order_px": float(meta.get("exit_order_px") or 0.0),
            "exit_order_age_ms": int(meta.get("exit_order_age_ms") or 0),
            "exit_requotes": int(meta.get("exit_requotes") or 0),
            "exit_patience_ms": int(meta.get("exit_patience_ms") or 0),
            "exit_requote_min_ms": int(meta.get("exit_requote_min_ms") or 0),

            "pos_delta_ticks": int(meta.get("pos_delta_ticks") or 0),
            # Phase 3.3a: explicit trailing activation inputs for offline replay
            "fee_ticks": int(meta.get("fee_ticks") or 0),
            "activate_ticks": int(meta.get("trail_activate_ticks") or 0),
            "peak_ticks": int(meta.get("trail_peak_ticks") or 0),
            "delta_ticks": int(meta.get("pos_delta_ticks") or 0),
        }

        live = {
            "autotune_offset_adj_ticks": int(meta.get("autotune_offset_adj_ticks") or 0),
            "autotune_entry_cooldown_ms": int(meta.get("autotune_entry_cooldown_ms") or 0),
            "autotune_requote_min_ms": int(meta.get("autotune_requote_min_ms") or 0),
            "safety_paused": bool(meta.get("safety_paused") or False),
            "safety_pause_left_ms": int(meta.get("safety_pause_left_ms") or 0),
            "safety_toxic_ms": int(meta.get("safety_toxic_ms") or 0),
        }

        ev = {
            "ts_ms": int(now_ms),
            "type": "frame",
            "symbol": str(getattr(self.s, "SYMBOL", "")),
            "book": book,
            "mark_price": float(m.get("mark_price") or 0.0),
            "mark_ts_ms": int(m.get("mark_ts_ms") or 0),
            "last_trade": last_trade,
            "tape": {"pressure": float(trade_pressure), "churn_ps": float(churn_ps)},
            "regime": {
                "state": str(meta.get("regime_state") or ""),
                "trend_strength": float(meta.get("regime_trend_strength") or 0.0),
                "chop_score": float(meta.get("regime_chop_score") or 0.0),
                "toxic_score": float(meta.get("regime_toxic_score") or 0.0),
                "eff_ratio": float(meta.get("regime_eff_ratio") or 0.0),
                "trend_ticks": float(meta.get("regime_trend_ticks") or 0.0),
                "spread_ticks": float(meta.get("regime_spread_ticks") or 0.0),
                "vol_ticks": float(meta.get("regime_vol_ticks") or 0.0),
                "spread_med": float(meta.get("regime_spread_med") or 0.0),
                "vol_med": float(meta.get("regime_vol_med") or 0.0),
                "toxic_hard": bool(meta.get("regime_toxic_hard") or False),
                "toxic_soft": bool(meta.get("regime_toxic_soft") or False),
                "countertrend": bool(meta.get("regime_countertrend") or False),
                "edge_mult": float(meta.get("regime_edge_mult") or 1.0),
                "offset_add_ticks": int(meta.get("regime_offset_add_ticks") or 0),
            },
            "swing": swing,
            "decision": decision,
            "live": live,
            "calib": {
                "h60": {
                    "n": int(meta.get("calib_60_n") or 0),
                    "bias_ticks": float(meta.get("calib_60_bias_ticks") or 0.0),
                    "mae_ticks": float(meta.get("calib_60_mae_ticks") or 0.0),
                    "hit_rate": float(meta.get("calib_60_hit_rate") or 0.0),
                },
                "h120": {
                    "n": int(meta.get("calib_120_n") or 0),
                    "bias_ticks": float(meta.get("calib_120_bias_ticks") or 0.0),
                    "mae_ticks": float(meta.get("calib_120_mae_ticks") or 0.0),
                    "hit_rate": float(meta.get("calib_120_hit_rate") or 0.0),
                },
                "h300": {
                    "n": int(meta.get("calib_300_n") or 0),
                    "bias_ticks": float(meta.get("calib_300_bias_ticks") or 0.0),
                    "mae_ticks": float(meta.get("calib_300_mae_ticks") or 0.0),
                    "hit_rate": float(meta.get("calib_300_hit_rate") or 0.0),
                },
            },
        }
        try:
            self.recorder.record(ev)
        except Exception:
            pass


    def _touch_liveness(self, now_ms: int) -> None:
        """Update internal loop liveness counters."""
        now_ms = int(now_ms)
        prev = int(getattr(self.state, "last_step_ts_ms", 0) or 0)
        self.state.loop_dt_ms = int(now_ms - prev) if prev > 0 else 0
        self.state.last_step_ts_ms = int(now_ms)

    def _gate_state_from_reason(self, reason: str) -> str:
        r = str(reason or "").upper()
        if not r:
            return ""
        if r.startswith("BOT_") or "KILL" in r:
            return "BOT"
        if "PAUSED" in r or r.startswith("PAUSE"):
            return "PAUSED"
        if "TOXIC" in r or "REGIME" in r:
            return "REGIME"
        if "EV_GATE" in r or r == "EV" or "GATED" == r:
            return "EV"
        if "COOLDOWN" in r:
            return "COOLDOWN"
        if "ORDER_YOUNG" in r or "MIN_ORDER" in r:
            return "ORDER_LIFE"
        if "DRIFT" in r:
            return "DRIFT"
        if "REQUOTE" in r:
            return "REQUOTE"
        if "SIGNAL" in r or r in ("NO_SIGNAL", "NO_DESIRED_SIDE"):
            return "SIGNAL"
        if "QTY_ZERO" in r or "CAP" in r or "SIZ" in r:
            return "SIZING"
        if r.startswith("IN_POS") or "EXIT" in r:
            return "POSITION"
        if "ENTRY" in r or "FLIP" in r:
            return "ENTRY"
        return "OTHER"

    async def _set_meta(self, snap: dict, now_ms: int, meta: dict) -> None:
        """Unified strategy_meta setter: stable schema + liveness keys."""
        conn = (snap.get("connectivity", {}) or {}) if isinstance(snap, dict) else {}
        ws_lag_ms = int(conn.get("ws_last_event_age_ms") or 0)

        reason = str((meta or {}).get("no_trade_reason") or "")
        gate_state = str((meta or {}).get("gate_state") or "") or self._gate_state_from_reason(reason)

        base = {
            "mode": str((meta or {}).get("mode") or str(getattr(self.s, "STRATEGY_MODE", "swing_1m") or "swing_1m")),
            "symbol": str((meta or {}).get("symbol") or getattr(self.s, "SYMBOL", "")),
            "no_trade_reason": reason,
            "gate_state": gate_state,
            "last_step_ts_ms": int(getattr(self.state, "last_step_ts_ms", 0) or int(now_ms)),
            "loop_dt_ms": int(getattr(self.state, "loop_dt_ms", 0) or 0),
            "heartbeat_ok": True,
            "ws_lag_ms": int(ws_lag_ms),
        }

        out = dict(base)
        try:
            out.update(dict(meta or {}))
        except Exception:
            pass
        # ensure these are never dropped
        out["no_trade_reason"] = reason
        out["gate_state"] = gate_state
        out["last_step_ts_ms"] = base["last_step_ts_ms"]
        out["loop_dt_ms"] = base["loop_dt_ms"]
        out["heartbeat_ok"] = True
        out["ws_lag_ms"] = int(ws_lag_ms)
        # Phase 2.1: recorder observability (dashboard)
        if self.recorder is not None:
            try:
                out.update(self.recorder.meta_dict())
            except Exception:
                pass
        await self.store.set_strategy_meta(out)

    def _trim_ts(self, dq: Deque[int], now_ms: int, win_ms: int) -> None:
        """Trim a deque of timestamps in-place to keep only [now-win, now]."""
        cutoff = int(now_ms) - int(win_ms)
        while dq and int(dq[0]) < cutoff:
            dq.popleft()

    def _percentile_int(self, values: Deque[int], p: float) -> int:
        """Approx percentile for small deques (OK for N<=400)."""
        if not values:
            return 0
        arr = sorted(int(v) for v in values)
        if not arr:
            return 0
        p = max(0.0, min(1.0, float(p)))
        k = int(round((len(arr) - 1) * p))
        k = max(0, min(len(arr) - 1, k))
        return int(arr[k])

    def _update_pnl_1h_events(self, now_ms: int, snap: dict) -> float:
        """Maintain rolling 1h realized pnl-after-fees from CLOSED trades list."""
        now_ms = int(now_ms)
        closed = (snap.get("closed_trades") or []) if isinstance(snap, dict) else []
        try:
            seen = int(getattr(self.state, "closed_seen", 0) or 0)
        except Exception:
            seen = 0
        if not isinstance(closed, list):
            closed = []

        if len(closed) > seen:
            for t in closed[seen:]:
                try:
                    ts = int(t.get("ts_close_ms") or now_ms)
                    net = float(t.get("net") or 0.0)
                    # net is expected to be after fees (realized - fees)
                    self.state.pnl_events_1h.append((ts, net))
                except Exception:
                    continue
            self.state.closed_seen = int(len(closed))

        # trim 1h
        cutoff = now_ms - 3_600_000
        while self.state.pnl_events_1h and int(self.state.pnl_events_1h[0][0]) < cutoff:
            self.state.pnl_events_1h.popleft()

        pnl_1h = 0.0
        for _, v in self.state.pnl_events_1h:
            pnl_1h += float(v)
        return float(pnl_1h)


    def _calib_bin_index(self, edge_ticks: float) -> int:
        try:
            v = float(edge_ticks)
        except Exception:
            return 0
        edges = getattr(self, "_calib_bin_edges", None) or []
        if not edges or len(edges) < 2:
            return 0
        for i in range(len(edges) - 1):
            lo = float(edges[i])
            hi = float(edges[i + 1])
            if v >= lo and v < hi:
                return int(i)
        return int(len(edges) - 2)

    def _process_pred_audit_deque(
        self,
        now_ms: int,
        mid_now: float,
        horizon_ms: int,
        dq: Deque[Tuple[int, float, float, int, float]],
        stats: _CalibStats,
        *,
        bins: Optional[List[_CalibBin]] = None,
    ) -> None:
        """Pop matured samples and update calibration stats.

        Sample tuple: (ts_ms, mid_at_pred, tick_size, dir, pred_edge_ticks)
        """
        now_ms = int(now_ms)
        while dq and (int(now_ms) - int(dq[0][0])) >= int(horizon_ms):
            try:
                ts0, mid0, tick0, dir0, pred_edge = dq.popleft()
                tick0 = float(tick0)
                if float(tick0) <= 0.0:
                    continue
                realized_dir = float(dir0) * (float(mid_now) - float(mid0)) / float(tick0)
                realized_fav = max(0.0, float(realized_dir))
                stats.update(float(pred_edge), float(realized_fav), float(realized_dir))
                if bins is not None:
                    bi = self._calib_bin_index(float(pred_edge))
                    if 0 <= int(bi) < len(bins):
                        bins[int(bi)].update(float(pred_edge), float(realized_fav), float(realized_dir))
            except Exception:
                # never break the trading loop due to audit math
                continue

    def _update_prediction_calibration(self, now_ms: int, *, mid: float, tick: float, pred: Any) -> Dict[str, Any]:
        """Phase 3.3: online prediction audit / calibration.

        Compares predicted edge_ticks_est to the realized *favorable* move in the predicted direction
        after 60/120/300 seconds.
        """
        out: Dict[str, Any] = {}
        if not bool(getattr(self, "_calib_enabled", True)):
            return out

        now_ms = int(now_ms)
        mid = float(mid)
        tick = float(tick)

        # Append new sample (downsampled)
        try:
            last_ms = int(getattr(self.state, "pred_audit_last_sample_ms", 0) or 0)
        except Exception:
            last_ms = 0
        sample_ms = int(getattr(self, "_calib_sample_ms", 250) or 250)

        try:
            dir0 = int(getattr(pred, "dir", 0) or 0)
            edge0 = float(getattr(pred, "edge_ticks_est", 0.0) or 0.0)
        except Exception:
            dir0 = 0
            edge0 = 0.0

        if mid > 0.0 and tick > 0.0 and int(dir0) != 0 and float(edge0) > 0.0:
            if (int(now_ms) - int(last_ms)) >= int(sample_ms):
                tpl = (int(now_ms), float(mid), float(tick), int(dir0), float(edge0))
                try:
                    self.state.pred_audit_60.append(tpl)
                    self.state.pred_audit_120.append(tpl)
                    self.state.pred_audit_300.append(tpl)
                except Exception:
                    pass
                self.state.pred_audit_last_sample_ms = int(now_ms)

        # Process matured samples even if no new sample was appended
        if mid > 0.0:
            self._process_pred_audit_deque(now_ms, mid, 60_000, self.state.pred_audit_60, self.state.calib_60)
            self._process_pred_audit_deque(now_ms, mid, 120_000, self.state.pred_audit_120, self.state.calib_120, bins=self._calib_bins_120)
            self._process_pred_audit_deque(now_ms, mid, 300_000, self.state.pred_audit_300, self.state.calib_300)

        s60 = self.state.calib_60.summary()
        s120 = self.state.calib_120.summary()
        s300 = self.state.calib_300.summary()

        def _fill(prefix: str, s: Dict[str, float]) -> None:
            out[f"{prefix}_n"] = int(s.get("n", 0) or 0)
            out[f"{prefix}_pred_mean_ticks"] = float(s.get("pred_mean", 0.0) or 0.0)
            out[f"{prefix}_real_fav_mean_ticks"] = float(s.get("real_fav_mean", 0.0) or 0.0)
            out[f"{prefix}_real_dir_mean_ticks"] = float(s.get("real_dir_mean", 0.0) or 0.0)
            out[f"{prefix}_bias_ticks"] = float(s.get("bias", 0.0) or 0.0)
            out[f"{prefix}_mae_ticks"] = float(s.get("mae", 0.0) or 0.0)
            out[f"{prefix}_rmse_ticks"] = float(s.get("rmse", 0.0) or 0.0)
            out[f"{prefix}_hit_rate"] = float(s.get("hit_rate", 0.0) or 0.0)

        _fill("calib_60", s60)
        _fill("calib_120", s120)
        _fill("calib_300", s300)

        # Bins (120s) for quick inspection in dashboard + offline logs.
        try:
            out["calib_bins_120"] = [b.to_dict() for b in (self._calib_bins_120 or [])]
        except Exception:
            out["calib_bins_120"] = []

        return out
    def _maker_price(self, side: str, bb: float, ba: float, tick: float) -> float:
        """Choose a maker-only price, optionally stepping inside the spread.

        If there is room (spread >= inside*tick), we place 1 tick (or N ticks)
        inside the spread to get higher fill probability without crossing.
        """
        eps = max(tick * 0.25, 1e-9)
        inside = max(0, int(self._peg_inside))

        if side == "BUY":
            base = float(bb)
            if inside <= 0:
                return base
            candidate = float(bb) + inside * float(tick)
            return candidate if candidate < float(ba) - eps else base

        # SELL
        base = float(ba)
        if inside <= 0:
            return base
        candidate = float(ba) - inside * float(tick)
        return candidate if candidate > float(bb) + eps else base

    async def _run(self) -> None:
        while not self._stop.is_set():
            try:
                # Default strategy is swing_1m; legacy scalp remains available.
                if str(self._strategy_mode).lower().startswith("swing"):
                    await self.swing_step()
                else:
                    await self._step()
            except Exception as e:
                log.exception("strategy_error: %s", e)
            # slightly faster loop = more responsive quoting on testnet
            await asyncio.sleep(0.10)


    async def swing_step(self) -> None:
        """1–5 minute swing strategy (maker-only, paper-only).

        Design goals:
          - trade less frequently than micro-scalp (fees-aware EV gate)
          - one-sided maker entry on pullbacks
          - TP/SL/Time-stop exits (reduce-side always allowed)
          - robust cancel/replace smoothing (no cancel-spam)
        """
        snap = await self.store.snapshot()
        now_ms = int(snap.get("ts_ms", self._now_ms()))

        # Phase 1.1: liveness counters for dashboard
        self._touch_liveness(int(now_ms))

        conn = snap.get("connectivity", {}) or {}
        bot_running = bool(conn.get("bot_running", False))
        kill_switch = bool(conn.get("kill_switch", False))
        kill_reason = str(conn.get("kill_reason", "") or "")

        # Phase 0: never be silent about why nothing happens.
        if not bot_running:
            reason = "BOT_STOPPED"
            meta = {
                "mode": "swing_1m",
                "symbol": str(getattr(self.s, "SYMBOL", "")),
                "no_trade_reason": reason,
            }
            await self._set_meta(snap, now_ms, meta)
            self._maybe_record_frame(now_ms, snap, pred=None, meta=meta)
            self._maybe_status_log(
                now_ms,
                mode="swing_1m",
                symbol=str(getattr(self.s, "SYMBOL", "")),
                bot_running=bot_running,
                kill_switch=kill_switch,
                reason=reason,
                in_pos=False,
                pos_qty=0.0,
                avg_px=0.0,
                open_orders=0,
                swing_dir=0,
                swing_strength=0.0,
                swing_edge_ticks=0.0,
                fee_ticks=0,
                min_profit_ticks=0,
                regime="",
                cooldown_left_ms=0,
            )
            return
        if kill_switch:
            reason = f"KILL:{kill_reason}" if kill_reason else "KILL_SWITCH"
            meta = {
                "mode": "swing_1m",
                "symbol": str(getattr(self.s, "SYMBOL", "")),
                "no_trade_reason": reason,
            }
            await self._set_meta(snap, now_ms, meta)
            self._maybe_record_frame(now_ms, snap, pred=None, meta=meta)
            self._maybe_status_log(
                now_ms,
                mode="swing_1m",
                symbol=str(getattr(self.s, "SYMBOL", "")),
                bot_running=bot_running,
                kill_switch=kill_switch,
                reason=reason,
                in_pos=False,
                pos_qty=0.0,
                avg_px=0.0,
                open_orders=0,
                swing_dir=0,
                swing_strength=0.0,
                swing_edge_ticks=0.0,
                fee_ticks=0,
                min_profit_ticks=0,
                regime="",
                cooldown_left_ms=0,
            )
            return

        mkt = snap.get("market", {}) or {}
        acct = snap.get("account", {}) or {}
        stats = snap.get("stats", {}) or {}
        orders = (snap.get("orders", []) or [])

        bb = float(mkt.get("best_bid") or 0.0)
        ba = float(mkt.get("best_ask") or 0.0)
        mid = float(mkt.get("mid") or ((bb + ba) / 2.0 if bb > 0 and ba > 0 else 0.0))
        spread = float(mkt.get("spread") or (ba - bb if ba > 0 and bb > 0 else 0.0))
        tick = float(mkt.get("tick_size") or 0.0)
        imb = float(mkt.get("imbalance") or 0.0)
        press = float(mkt.get("trade_pressure") or 0.0)
        vol = float(mkt.get("volatility") or 0.0)
        mark = float(mkt.get("mark_price") or mid)

        prev_pos_qty = float(getattr(self.state, "last_pos_qty", 0.0) or 0.0)
        pos = float(acct.get("position_qty") or 0.0)
        avg = float(acct.get("avg_entry_price") or 0.0)
        in_pos = abs(pos) > 1e-12

        # Phase 1.5: maintain rolling 1h pnl-after-fees from closed trades
        pnl_1h_after_fees = float(self._update_pnl_1h_events(now_ms, snap))

        # Phase 1.5: entry fill detection (flat -> in_pos transition)
        if in_pos and abs(float(prev_pos_qty)) < 1e-12:
            self.state.entry_fill_ts.append(int(now_ms))
            if int(getattr(self.state, "pending_entry_ts_ms", 0) or 0) > 0:
                dt = int(now_ms) - int(self.state.pending_entry_ts_ms)
                if dt >= 0:
                    self.state.time_to_fill_ms.append(int(dt))
            self.state.pending_entry_ts_ms = 0

        # Open orders (maker-only) bookkeeping
        open_orders = [o for o in orders if str(o.get("status", "")) in ("NEW", "PARTIALLY_FILLED")]

        # --- Fee economics (re-use existing logic) ---
        fee_bps = float(getattr(self.s, "MAKER_FEE_BPS", 0.0) or 0.0)
        fee_rate = max(0.0, fee_bps / 10000.0)
        fee_ticks = 0
        if mid > 0 and tick > 0 and fee_rate > 0:
            fee_ticks = int(math.ceil((2.0 * fee_rate * mid) / tick))
        fee_ticks = max(0, int(fee_ticks))
        min_profit_ticks = max(0, int(fee_ticks) + int(getattr(self.s, "STRATEGY_FEE_BUFFER_TICKS", 1)))

        # --- Regime tracking (rolling window) ---
        reg = self._update_regime(now_ms, float(bb), float(ba), float(tick), float(mid), float(imb))
        churn_ps = float(reg.get("churn_ps", 0.0) or 0.0)
        imb_std = float(reg.get("imb_std", 0.0) or 0.0)
        trend_ticks_5s = float(reg.get("trend_ticks", 0.0) or 0.0)

        # NOTE: Regime v2 classification (trend/chop/toxic) is computed after SwingSignal
        # features are available (spread/vol in ticks). We keep these base metrics for diagnostics.


        # --- Swing signal ---
        last_trade_px = 0.0
        try:
            trades = (snap.get("trades") or []) if isinstance(snap, dict) else []
            if isinstance(trades, list) and trades:
                last_trade_px = float((trades[0] or {}).get("price") or 0.0)
        except Exception:
            last_trade_px = 0.0

        self._swing.observe(
            now_ms,
            mid=float(mid),
            tick_size=float(tick),
            spread=float(spread),
            imbalance=float(imb),
            trade_pressure=float(press),
            churn_proxy=float(churn_ps),
            mark_price=float(mark),
            last_trade_price=float(last_trade_px),
        )
        pred = self._swing.predict(now_ms, mid=float(mid), tick_size=float(tick))

        swing_dir = int(pred.dir)
        swing_strength = float(pred.strength)
        swing_edge_ticks = float(pred.edge_ticks_est)
        swing_score_ticks = float(pred.score_ticks)
        spread_ticks = float(pred.spread_ticks)
        atr_ticks_120 = float(getattr(pred, "atr_ticks_120", 0.0) or 0.0)
        slope_norm = float(getattr(pred, "slope_norm", 0.0) or 0.0)
        imb_ema = float(getattr(pred, "imbalance_ema", 0.0) or 0.0)
        imb_std_sig = float(getattr(pred, "imb_std", 0.0) or 0.0)
        mark_last_div_ticks = float(getattr(pred, "mark_last_div_ticks", 0.0) or 0.0)
        vol_ticks_old = float(pred.vol_abs_ret_ema) * float(mid) / float(tick) if mid > 0 and tick > 0 else 0.0
        vol_ticks = float(atr_ticks_120) if float(atr_ticks_120) > 0 else float(vol_ticks_old)

        # --- Phase 3.1: regime classifier v2 (trend/chop/toxic) ---
        reg2 = self._classify_regime_v2(
            now_ms,
            tick=float(tick),
            mid=float(mid),
            spread_ticks=float(spread_ticks),
            vol_ticks=float(vol_ticks),
            churn_ps=float(churn_ps),
        )
        regime_class = str(reg2.get("state", "NORMAL") or "NORMAL")
        toxic_hard = bool(reg2.get("toxic_hard", False))
        toxic_soft = bool(reg2.get("toxic_soft", False))
        regime_trend_strength = float(reg2.get("trend_strength", 0.0) or 0.0)
        regime_chop_score = float(reg2.get("chop_score", 0.0) or 0.0)
        regime_toxic_score = float(reg2.get("toxic_score", 0.0) or 0.0)
        regime_eff_ratio = float(reg2.get("eff_ratio", 0.0) or 0.0)
        trend_ticks_v2 = float(reg2.get("trend_ticks", 0.0) or 0.0)
        regime_spread_med = float(reg2.get("spread_med", 0.0) or 0.0)
        regime_vol_med = float(reg2.get("vol_med", 0.0) or 0.0)

        # soft guards: tighten EV gate / offset without total stop (hard toxic can block)
        countertrend = bool(
            (regime_class in ("TREND_UP", "TREND_DOWN"))
            and ((regime_class == "TREND_UP" and int(swing_dir) < 0) or (regime_class == "TREND_DOWN" and int(swing_dir) > 0))
        )

        regime_edge_mult = 1.0
        regime_offset_add_ticks = 0
        if countertrend:
            regime_edge_mult *= float(getattr(self.s, "STRATEGY_REGIME_V2_COUNTERTREND_EDGE_MULT", 1.25) or 1.25)
            regime_offset_add_ticks += int(getattr(self.s, "STRATEGY_REGIME_V2_COUNTERTREND_OFFSET_ADD_TICKS", 1) or 1)
        if regime_class == "CHOP":
            regime_edge_mult *= float(getattr(self.s, "STRATEGY_REGIME_V2_CHOP_EDGE_MULT", 1.10) or 1.10)
            regime_offset_add_ticks += int(getattr(self.s, "STRATEGY_REGIME_V2_CHOP_OFFSET_ADD_TICKS", 0) or 0)
        if toxic_soft:
            regime_edge_mult *= float(getattr(self.s, "STRATEGY_REGIME_V2_TOXIC_SOFT_EDGE_MULT", 1.15) or 1.15)
            regime_offset_add_ticks += int(getattr(self.s, "STRATEGY_REGIME_V2_TOXIC_SOFT_OFFSET_ADD_TICKS", 2) or 2)

        # cap offset adds (soft guards must stay bounded)
        try:
            max_add = int(getattr(self.s, "STRATEGY_REGIME_V2_OFFSET_ADD_MAX_TICKS", 6) or 6)
            regime_offset_add_ticks = max(0, min(int(regime_offset_add_ticks), int(max_add)))
        except Exception:
            regime_offset_add_ticks = max(0, int(regime_offset_add_ticks))

        # Autotuner safety sees only hard toxic to avoid over-pausing
        regime_for_autotune = "TOXIC" if toxic_hard else ""


        # --- Phase 3.3: prediction audit / calibration (online) ---
        calib_meta = self._update_prediction_calibration(int(now_ms), mid=float(mid), tick=float(tick), pred=pred)

        # --- Phase 1.5: rolling execution metrics + bounded autotuner (ENTRY only) ---
        # Trim windows
        self._trim_ts(self.state.entry_attempt_ts, now_ms, 300_000)
        self._trim_ts(self.state.entry_fill_ts, now_ms, 300_000)
        self._trim_ts(self.state.loop_ts_5m, now_ms, 300_000)
        self._trim_ts(self.state.ev_gate_open_ts, now_ms, 300_000)

        # Count this loop for EV-gate open rate denominator
        self.state.loop_ts_5m.append(int(now_ms))
        self._trim_ts(self.state.loop_ts_5m, now_ms, 300_000)

        ttf_p50 = int(self._percentile_int(self.state.time_to_fill_ms, 0.50))
        ttf_p90 = int(self._percentile_int(self.state.time_to_fill_ms, 0.90))
        ttf_n = int(len(self.state.time_to_fill_ms))

        loop_n_pre = max(1, int(len(self.state.loop_ts_5m)))
        ev_open_rate_pre = float(len(self.state.ev_gate_open_ts)) / float(loop_n_pre)

        attempts_5m = int(len(self.state.entry_attempt_ts))
        fills_5m = int(len(self.state.entry_fill_ts))
        fill_rate_5m = float(fills_5m) / float(max(1, attempts_5m))

        rejects_5m = int(stats.get("rejects_5m", 0) or 0)
        cancels_5m = int(stats.get("cancels_5m", 0) or 0)
        reject_rate_5m = float(stats.get("reject_rate_5m", 0.0) or 0.0)
        cancel_rate_5m = float(stats.get("cancel_rate_5m", 0.0) or 0.0)
        fees_1h = float(stats.get("fees_1h", 0.0) or 0.0)
        equity_usdt = float(acct.get("equity", 0.0) or 0.0)

        metrics_pre = ExecMetrics(
            entry_attempts_5m=attempts_5m,
            entry_fills_5m=fills_5m,
            fill_rate_5m=float(fill_rate_5m),
            rejects_5m=rejects_5m,
            cancels_5m=cancels_5m,
            reject_rate_5m=float(reject_rate_5m),
            cancel_rate_5m=float(cancel_rate_5m),
            time_to_fill_p50_ms=ttf_p50,
            time_to_fill_p90_ms=ttf_p90,
            time_to_fill_n=ttf_n,
            ev_gate_open_rate_5m=float(ev_open_rate_pre),
            pnl_1h_after_fees=float(pnl_1h_after_fees),
            fees_1h=float(fees_1h),
        )

        live_params, autotune_action, autotune_reason = self._autotune.update(
            now_ms=int(now_ms),
            metrics=metrics_pre,
            regime=str(regime_for_autotune),
            equity_usdt=float(equity_usdt),
        )

        # Effective gate: live multipliers (autotune) * regime soft guards
        min_edge_mult_effective = float(live_params.min_edge_mult_live) * float(regime_edge_mult)
        try:
            cap = float(getattr(self.s, "STRATEGY_REGIME_V2_EDGE_MULT_CAP", 3.0) or 3.0)
            min_edge_mult_effective = min(float(min_edge_mult_effective), float(cap))
        except Exception:
            min_edge_mult_effective = float(min_edge_mult_effective)

        # Now compute EV-gate open tick for this loop using effective gate
        ev_gate_open_now = bool(
            int(swing_dir) != 0
            and (not bool(toxic_hard))
            and float(swing_edge_ticks) >= float(min_profit_ticks) * float(min_edge_mult_effective)
        )
        if ev_gate_open_now:
            self.state.ev_gate_open_ts.append(int(now_ms))
        self._trim_ts(self.state.ev_gate_open_ts, now_ms, 300_000)

        loop_n = max(1, int(len(self.state.loop_ts_5m)))
        ev_open_rate_5m = float(len(self.state.ev_gate_open_ts)) / float(loop_n)

        # Phase 1.5: common meta fields (always include in dashboard)
        pause_left_ms = 0
        try:
            if not bool(live_params.trading_enabled):
                pause_left_ms = max(0, int(live_params.pause_until_ms) - int(now_ms))
        except Exception:
            pause_left_ms = 0

        phase15_meta_base = {
            # rolling execution (entry-only)
            "entry_attempts_5m": int(attempts_5m),
            "entry_fills_5m": int(fills_5m),
            "fill_rate_5m": float(fill_rate_5m),
            "time_to_fill_p50_ms": int(ttf_p50),
            "time_to_fill_p90_ms": int(ttf_p90),
            "ev_gate_open_rate_5m": float(ev_open_rate_5m),
            "pnl_1h_after_fees": float(pnl_1h_after_fees),

            # autotune (bounded)
            "autotune_enabled": bool(getattr(self._autotune, "enabled", False)),
            "autotune_last_tune_ms": int(getattr(self._autotune, "last_tune_ms", 0) or 0),
            "autotune_action": str(autotune_action or ""),
            "autotune_reason": str(autotune_reason or ""),
            "autotune_offset_adj_ticks": int(live_params.offset_adj_ticks),
            "autotune_entry_cooldown_ms": int(live_params.entry_cooldown_ms),
            "autotune_requote_min_ms": int(live_params.requote_min_ms),
            "min_edge_mult_live": float(live_params.min_edge_mult_live),

            # regime classifier v2 (Phase 3.1)
            "regime_state": str(regime_class),
            "regime_trend_strength": float(regime_trend_strength),
            "regime_chop_score": float(regime_chop_score),
            "regime_toxic_score": float(regime_toxic_score),
            "regime_eff_ratio": float(regime_eff_ratio),
            "regime_trend_ticks": float(trend_ticks_v2),
            "regime_spread_ticks": float(spread_ticks),
            "regime_vol_ticks": float(vol_ticks),
            "regime_spread_med": float(regime_spread_med),
            "regime_vol_med": float(regime_vol_med),
            "regime_toxic_hard": bool(toxic_hard),
            "regime_toxic_soft": bool(toxic_soft),
            "regime_countertrend": bool(countertrend),
            "regime_edge_mult": float(regime_edge_mult),
            "regime_offset_add_ticks": int(regime_offset_add_ticks),
            "min_edge_mult_effective": float(min_edge_mult_effective),

            # Phase 3.1b: regime effective guards (observable)
            "regime_effective_min_edge_mult": float(min_edge_mult_effective),
            "regime_effective_offset_add_ticks": int(regime_offset_add_ticks),
            "regime_entry_allowed": bool(not bool(toxic_hard)),

            # safety pause (entries only)
            "trading_enabled": bool(live_params.trading_enabled),
            "pause_until_ms": int(live_params.pause_until_ms),
            "pause_reason": str(live_params.pause_reason or ""),
            "pause_left_ms": int(pause_left_ms),
            "trail_dd_pct": float(self._swing_trail_dd_pct),
            "trail_k": float(self._swing_trail_atr_k),
        }
        try:
            phase15_meta_base.update(dict(calib_meta or {}))
        except Exception:
            pass

        # Analyzer input metrics (diagnostics only)
        analyzer_metrics = {
            "fill_rate_5m": float(fill_rate_5m),
            "reject_rate_5m": float(reject_rate_5m),
            "cancel_rate_5m": float(cancel_rate_5m),
            "ev_gate_open_rate_5m": float(ev_open_rate_5m),
            "fee_ticks": int(fee_ticks),
            "min_profit_ticks": int(min_profit_ticks),
            "spread_ticks": float(spread_ticks),
            "vol_ticks": float(vol_ticks),
            "swing_edge_ticks": float(swing_edge_ticks),
            "min_edge_mult_live": float(live_params.min_edge_mult_live),
            "min_edge_mult_effective": float(min_edge_mult_effective),

            # Phase 3.1b: regime effective guards (observable)
            "regime_effective_min_edge_mult": float(min_edge_mult_effective),
            "regime_effective_offset_add_ticks": int(regime_offset_add_ticks),
            "regime_entry_allowed": bool(not bool(toxic_hard)),
            "regime_state": str(regime_class),
            "regime_trend_strength": float(regime_trend_strength),
            "regime_chop_score": float(regime_chop_score),
            "regime_toxic_score": float(regime_toxic_score),
            "regime_eff_ratio": float(regime_eff_ratio),
            "regime_edge_mult": float(regime_edge_mult),
            "regime_offset_add_ticks": int(regime_offset_add_ticks),
            "toxic_hard": bool(toxic_hard),
            "toxic_soft": bool(toxic_soft),
            "countertrend": bool(countertrend),
            "pause_left_ms": int(pause_left_ms),
        }

        # helper: cancel an order id
        async def _cancel_order_id(oid: str) -> None:
            if oid:
                try:
                    await self.engine.cancel_order(str(oid))
                except Exception:
                    pass

        # helper: cancel all open orders not matching a side
        async def _cancel_non_side(keep_side: str) -> None:
            for o in open_orders:
                side = str(o.get("side", ""))
                if side != keep_side:
                    oid = str(o.get("order_id") or o.get("id") or "")
                    await _cancel_order_id(oid)

        # --- Position management (reduce-side always allowed) ---
        tp_ticks = max(int(self._swing_tp_min_ticks), int(min_profit_ticks * float(self._swing_tp_mult_fees)))
        sl_ticks = max(int(self._swing_sl_min_ticks), int(min_profit_ticks * float(self._swing_sl_mult_fees)))
        max_hold_ms = int(self._swing_max_hold_ms)

        # Detect position open time
        if in_pos and abs(float(self.state.last_pos_qty)) < 1e-12:
            self.state.pos_open_ts_ms = int(now_ms)
            # Reset trailing state at new position open
            # Reset exit execution state at new position open
            self.state.swing_exit_started_ms = 0
            self.state.swing_exit_last_requote_ms = 0
            self.state.swing_exit_requotes = 0
            self.state.swing_trail_active = False
            self.state.swing_trail_peak_ticks = 0
            self.state.swing_trail_retrace_ticks = 0
            self.state.swing_trail_activate_ticks = 0
            self.state.swing_trail_mode = ""
            self.state.swing_trail_be_floor_ticks = 0
            self.state.swing_trail_letrun_on = False
        if not in_pos:
            self.state.pos_open_ts_ms = 0
            self.state.swing_exit_reason = ""
            self.state.swing_exit_started_ms = 0
            self.state.swing_exit_last_requote_ms = 0
            self.state.swing_exit_requotes = 0
            self.state.swing_trail_active = False
            self.state.swing_trail_peak_ticks = 0
            self.state.swing_trail_retrace_ticks = 0
            self.state.swing_trail_activate_ticks = 0
            self.state.swing_trail_mode = ""
            self.state.swing_trail_be_floor_ticks = 0
            self.state.swing_trail_letrun_on = False
        self.state.last_pos_qty = float(pos)

        hold_ms = int(now_ms) - int(self.state.pos_open_ts_ms or now_ms) if in_pos else 0

        exit_reason = ""
        exit_side = ""
        desired_exit_px = 0.0
        chase_exit = False

        # Phase 3.3d: anti-loser controls (computed every loop; state remains maker-only)
        early_as_triggered = False
        time_decay_active = False
        time_decay_level = 0
        # Exit aggressiveness: inside ticks for maker-only exits (0..spread-1 bounded by helper)
        exit_inside_ticks = 1  # default "slightly inside" (0–1 tick) for better fill odds

        # For observability: keep these defined even if we can't compute mark/avg.
        delta_ticks = 0
        trail_trigger_level_ticks = 0

        if in_pos and avg > 0 and tick > 0 and mark > 0:
            # Profit in ticks (positive = favorable)
            delta = (mark - avg) if pos > 0 else (avg - mark)
            delta_ticks = int(round(delta / tick)) if tick > 0 else 0

            # Phase 3.3d (part 1): early adverse-selection exit (anti-knife)
            # If we get hit into a position and it immediately runs against us in toxic/churn,
            # try to exit earlier (maker-only) instead of waiting for full SL/time-stop.
            if (
                (not exit_reason)
                and bool(self._swing_early_as_enable)
                and int(hold_ms) <= int(self._swing_early_as_window_ms)
                and int(delta_ticks) <= -int(self._swing_early_as_loss_ticks)
                and (
                    float(regime_toxic_score) >= float(self._swing_early_as_toxic_score)
                    or float(churn_ps) >= float(self._swing_early_as_churn_ps)
                )
            ):
                early_as_triggered = True
                exit_reason = "EARLY_AS"
                exit_side = "SELL" if pos > 0 else "BUY"
                chase_exit = True
                # slightly more aggressive maker exit (bounded by spread in helper)
                exit_inside_ticks = max(int(exit_inside_ticks), 1 + max(0, int(self._swing_early_as_inside_add)))

            # Phase 3.3d (part 2): time-decay exits for losers
            # As the position ages *while still losing*, we gradually become more aggressive with maker exit
            # (move further inside the spread), while respecting min_order_life + requote_min_ms.
            if (
                bool(self._swing_time_decay_enable)
                and int(delta_ticks) < 0
                and int(hold_ms) >= int(self._swing_time_decay_start_ms)
                and abs(int(delta_ticks)) >= int(self._swing_time_decay_min_loss_ticks)
            ):
                time_decay_active = True
                step = max(1, int(self._swing_time_decay_step_ms))
                lvl = 1 + max(0, (int(hold_ms) - int(self._swing_time_decay_start_ms)) // int(step))
                lvl = self._clamp_int(int(lvl), 1, max(1, int(self._swing_time_decay_max_level)))
                time_decay_level = int(lvl)

                add = int(lvl) * max(0, int(self._swing_time_decay_inside_add_per_level))
                add = self._clamp_int(int(add), 0, max(0, int(self._swing_time_decay_max_inside_add)))
                exit_inside_ticks = max(int(exit_inside_ticks), 1 + int(add))

                # If nothing else already triggers an exit (and we're not already beyond SL), time-decay becomes the exit reason.
                # If delta is already beyond SL, we let SL label it (but keep time_decay_active + exit_inside_ticks).
                if (not exit_reason) and (int(delta_ticks) > -int(sl_ticks)):
                    exit_reason = "TIME_DECAY"
                    exit_side = "SELL" if pos > 0 else "BUY"
                    chase_exit = True

            # Phase 3.3a: trailing profit activation (fee-cover is NOT TP)
            # - trailing is a *profit-management* mode that starts only after a meaningful move
            # - activation threshold is based on fee_ticks * A, with an absolute floor (ETH default 8)
            if (not exit_reason) and bool(self._swing_trail_enable) and int(fee_ticks) > 0:
                act_ticks = max(
                    int(math.ceil(float(fee_ticks) * float(self._swing_trail_activate_mult_fees))),
                    int(self._swing_trail_min_activate_ticks),
                )
                self.state.swing_trail_activate_ticks = int(act_ticks)

                # Activate/update peak only for favorable deltas
                if int(delta_ticks) >= int(act_ticks) and int(delta_ticks) > 0:
                    if not bool(self.state.swing_trail_active):
                        self.state.swing_trail_active = True
                        self.state.swing_trail_peak_ticks = int(delta_ticks)
                    else:
                        self.state.swing_trail_peak_ticks = max(int(self.state.swing_trail_peak_ticks), int(delta_ticks))

                peak = int(self.state.swing_trail_peak_ticks)
                if bool(self.state.swing_trail_active) and peak > 0:
                    # Phase 3.3c: two-mode trailing
                    #   PROTECT: once profit is meaningful, keep at least breakeven+ (fees * BE_MULT + buffer)
                    #   LET_RUN: after peak grows or time passes, widen the trail (larger K) to avoid noise exits
                    be_floor_ticks = int(math.ceil(float(fee_ticks) * float(self._swing_trail_be_mult))) + int(self._swing_trail_be_buffer_ticks)
                    self.state.swing_trail_be_floor_ticks = int(be_floor_ticks)

                    letrun_on = bool(peak >= (int(act_ticks) * 2) or int(hold_ms) >= int(self._swing_trail_letrun_after_ms))
                    self.state.swing_trail_letrun_on = bool(letrun_on)
                    self.state.swing_trail_mode = "LET_RUN" if letrun_on else "PROTECT"

                    k_eff = float(self._swing_trail_atr_k) * (float(self._swing_trail_letrun_k_mult) if letrun_on else 1.0)
                    atr_part = int(math.ceil(float(atr_ticks_120) * float(k_eff))) if float(atr_ticks_120) > 0 else 0

                    retrace = max(
                        int(math.ceil(float(peak) * float(self._swing_trail_dd_pct))),
                        int(self._swing_trail_min_retrace_ticks),
                        int(atr_part),
                    )
                    self.state.swing_trail_retrace_ticks = int(retrace)

                    trail_trigger_level_ticks = max(0, int(peak) - int(retrace))

                    # Protect floor only applies after we have *reached* it at least once (avoid immediate exits).
                    if int(be_floor_ticks) > 0 and peak >= int(be_floor_ticks):
                        trail_trigger_level_ticks = max(int(trail_trigger_level_ticks), int(be_floor_ticks))

                    # Trigger only if peak was meaningful
                    min_peak = max(int(self._swing_trail_min_peak_ticks), int(act_ticks))
                    if peak >= int(min_peak) and int(delta_ticks) <= int(trail_trigger_level_ticks):
                        exit_reason = "TRAIL"
                        exit_side = "SELL" if pos > 0 else "BUY"
                        chase_exit = True

            # SL should dominate other loser exits (EARLY_AS / TIME_DECAY) if we breach max loss.
            if delta_ticks <= -int(sl_ticks):
                exit_reason = "SL"
                exit_side = "SELL" if pos > 0 else "BUY"
                chase_exit = True
            elif hold_ms >= int(max_hold_ms) and str(exit_reason or "") not in ("SL", "TRAIL", "TP", "EMERGENCY"):
                exit_reason = "TIME"
                exit_side = "SELL" if pos > 0 else "BUY"
                chase_exit = True
            elif (not exit_reason) and (not bool(self._swing_trail_enable)) and delta_ticks >= int(tp_ticks):
                # Legacy fee-aware TP (kept for compatibility if trailing is disabled)
                exit_reason = "TP"
                exit_side = "SELL" if pos > 0 else "BUY"
                desired_exit_px = (avg + tp_ticks * tick) if pos > 0 else (avg - tp_ticks * tick)
                chase_exit = False

        if in_pos:
            # Cancel any entry/other-side orders (avoid fighting unwind)
            if exit_side:
                await _cancel_non_side(exit_side)

            else:
                # Not exiting: clear exit window state (prevents stale counters)
                self.state.swing_exit_started_ms = 0
                self.state.swing_exit_last_requote_ms = 0
                self.state.swing_exit_requotes = 0

            desired_qty = abs(float(pos))

            # Find existing open order on exit side (pick oldest to avoid cancel loops)
            exit_orders = [o for o in open_orders if str(o.get("side", "")) == exit_side] if exit_side else []
            existing = min(exit_orders, key=lambda x: int(x.get("ts_ms", 0) or 0)) if exit_orders else None

            # Determine target price for reduce order
            eps = max(float(tick) * 0.25, 1e-9)
            if chase_exit:
                # Maker-only exit: sit at touch or slightly inside the spread (0–1 tick).
                # This reduces "far" exits that never fill while still preserving post-only.
                target_px = self._maker_price_inside(exit_side, float(bb), float(ba), float(tick), int(exit_inside_ticks))
            else:
                target_px = float(desired_exit_px) if desired_exit_px > 0 else 0.0
                # Ensure maker-only constraints
                if exit_side == "SELL" and bb > 0:
                    target_px = max(target_px, bb + eps)
                if exit_side == "BUY" and ba > 0:
                    target_px = min(target_px, ba - eps)

            exit_order_px = 0.0
            exit_order_age_ms = 0
            # Place/adjust reduce order
            if exit_side and target_px > 0 and desired_qty > 0:
                # Phase 3.4: exit execution discipline
                # - default: rest at touch (maker-only)
                # - wait exit_patience_ms before repricing
                # - at most one repricing step every exit_requote_min_ms
                exit_patience_ms = int(self._swing_exit_patience_ms)
                exit_requote_min_ms = int(self._swing_exit_requote_min_ms)
                exit_rq_drift_ticks = float(self._swing_exit_rq_drift_ticks)

                # Track an "exit window" so we don't cancel/replace in a loop
                if int(self.state.swing_exit_started_ms or 0) == 0:
                    self.state.swing_exit_started_ms = int(now_ms)
                    self.state.swing_exit_last_requote_ms = 0
                    self.state.swing_exit_requotes = 0

                if not existing:
                    o = await self.engine.place_limit_post_only(exit_side, price=float(target_px), qty=float(desired_qty))
                    exit_order_px = float(getattr(o, "price", target_px) or target_px)
                    log.info(
                        "exit_place side=%s reason=%s px=%.2f qty=%.6f patience_ms=%d rq_min_ms=%d",
                        str(exit_side),
                        str(exit_reason or ""),
                        float(exit_order_px),
                        float(desired_qty),
                        int(exit_patience_ms),
                        int(exit_requote_min_ms),
                    )
                    exit_order_age_ms = max(0, int(now_ms) - int(getattr(o, "ts_ms", now_ms) or now_ms))
                else:
                    oid = str(existing.get("order_id") or existing.get("id") or "")
                    o_px = float(existing.get("price", 0.0) or 0.0)
                    o_ts = int(existing.get("ts_ms", 0) or 0)
                    age_ms = int(now_ms) - int(o_ts)
                    drift_ticks = abs(float(target_px) - float(o_px)) / float(tick) if tick > 0 else 0.0

                    exit_order_px = float(o_px)
                    exit_order_age_ms = int(age_ms)

                    # If qty changed materially (partial fills), refresh once it's not too fresh
                    qty_need_refresh = abs(float(existing.get("remaining_qty", existing.get("qty", 0.0) or 0.0)) - float(desired_qty)) / max(desired_qty, 1e-9) > 0.15

                    # Respect patience (since exit window start) and throttled repricing
                    exit_window_age_ms = int(now_ms) - int(self.state.swing_exit_started_ms or now_ms)
                    min_life_ms = max(1500, int(self._swing_min_order_life_ms))

                    can_rq = (int(now_ms) - int(self.state.swing_exit_last_requote_ms or 0)) >= int(exit_requote_min_ms)
                    patient_enough = int(exit_window_age_ms) >= int(exit_patience_ms)

                    need = False
                    if qty_need_refresh and age_ms >= int(min_life_ms):
                        need = True
                    if chase_exit and patient_enough and age_ms >= int(min_life_ms) and drift_ticks >= float(exit_rq_drift_ticks):
                        need = True

                    if need and can_rq and oid:
                        await _cancel_order_id(oid)
                        self.state.swing_exit_last_requote_ms = int(now_ms)
                        self.state.swing_exit_requotes = int(self.state.swing_exit_requotes or 0) + 1
                        o = await self.engine.place_limit_post_only(exit_side, price=float(target_px), qty=float(desired_qty))
                        exit_order_px = float(getattr(o, "price", target_px) or target_px)
                        exit_order_age_ms = 0 # Optional emergency exit if time-stop persists
            if (
                in_pos
                and exit_reason == "TIME"
                and bool(self._swing_emergency)
                and hold_ms >= int(max_hold_ms) + 20_000
            ):
                # Cancel any resting orders first
                for o in list(open_orders):
                    oid = str(o.get("order_id") or o.get("id") or "")
                    await _cancel_order_id(oid)
                # Emergency fill at mark
                try:
                    await self.engine.emergency_exit(exit_side, price=float(mark), qty=abs(float(pos)), reason="TIME")
                    exit_reason = "EMERGENCY"
                except Exception:
                    pass

            self.state.swing_exit_reason = str(exit_reason or "")
            no_trade_reason = (
                (f"IN_POS_EXIT_{exit_reason}" if exit_reason else "IN_POS_HOLD")
            )
            # Update dashboard meta (position)
            equity = float(acct.get("equity_usdt", 0.0) or 0.0)
            max_notional = max(0.0, float(equity) * float(getattr(self.s, "STRATEGY_MAX_NOTIONAL_PCT", 0.10)))
            meta = {
                "mode": "swing_1m",
                "symbol": str(getattr(self.s, "SYMBOL", "")),
                "no_trade_reason": str(no_trade_reason),
                "regime_state": str(regime_class),
                "size_mult": 1.0,
                "base_seed_qty": float(abs(pos)),
                "base_notional_pct": 0.0,
                "side_mult_buy": 1.0,
                "side_mult_sell": 1.0,
                "dd_now": 0.0,
                "side_gate_buy": 1.0,
                "side_gate_sell": 1.0,
                "side_inside_adj_buy": 0,
                "side_inside_adj_sell": 0,
                "side_pnl_avg_buy": 0.0,
                "side_pnl_avg_sell": 0.0,
                "side_trades_buy": 0,
                "side_trades_sell": 0,
                "side_fq_n": 0,
                "fee_ticks": int(fee_ticks),
                "min_profit_ticks": int(min_profit_ticks),
                "min_inside_ticks": int(min_profit_ticks),
                "gate_ticks": int(math.ceil(float(min_profit_ticks) * float(min_edge_mult_effective))),
                "inside_used": 0,
                "swing_dir": int(swing_dir),
                "swing_strength": float(swing_strength),
                "swing_edge_ticks": float(swing_edge_ticks),
                "swing_score_ticks": float(swing_score_ticks),
                "atr_ticks_120": float(atr_ticks_120),
                "slope_norm": float(slope_norm),
                "imb_ema": float(imb_ema),
                "imb_std": float(imb_std_sig),
                "mark_last_div_ticks": float(mark_last_div_ticks),
                "trade_pressure": float(press),
                "ml_p_up": 0.5,
                "ml_edge": 0.0,
                "ml_h": 0,
                "ml_n": 0,
                "bandit_arm": "",
                "bandit_eps": 0.0,
                "entry_offset_ticks": 0,
                "entry_price": 0.0,
                "tp_ticks": int(tp_ticks),
                "sl_ticks": int(sl_ticks),
                # trailing profit (Phase 3.1b)
                "trail_enabled": bool(self._swing_trail_enable),
                "trail_active": bool(self.state.swing_trail_active),
                "trail_activate_ticks": int(self.state.swing_trail_activate_ticks),
                "trail_peak_ticks": int(self.state.swing_trail_peak_ticks),
                "trail_retrace_ticks": int(self.state.swing_trail_retrace_ticks),
                "trail_trigger_level_ticks": int(trail_trigger_level_ticks),
                "trail_dist_ticks": int(self.state.swing_trail_retrace_ticks),
                "trail_trigger_ticks": int(trail_trigger_level_ticks),
                "trail_dd_pct": float(self._swing_trail_dd_pct),
                "trail_k": float(self._swing_trail_atr_k),
                "trail_atr_ticks": float(atr_ticks_120),
                "trail_mode": str(self.state.swing_trail_mode or ""),
                "be_floor_ticks": int(self.state.swing_trail_be_floor_ticks),
                "letrun_on": bool(self.state.swing_trail_letrun_on),
                "letrun_after_ms": int(self._swing_trail_letrun_after_ms),

                # Phase 3.3d: anti-loser observability
                "early_as_triggered": bool(early_as_triggered),
                "early_as_loss_ticks": int(self._swing_early_as_loss_ticks),
                "early_as_window_ms": int(self._swing_early_as_window_ms),
                "time_decay_active": bool(time_decay_active),
                "time_decay_level": int(time_decay_level),
                "exit_inside_ticks": int(exit_inside_ticks),

                # Phase 3.4: exit execution discipline observability
                "exit_order_px": float(exit_order_px),
                "exit_order_age_ms": int(exit_order_age_ms),
                "exit_requotes": int(self.state.swing_exit_requotes or 0),
                "exit_patience_ms": int(self._swing_exit_patience_ms),
                "exit_requote_min_ms": int(self._swing_exit_requote_min_ms),

                "pos_delta_ticks": int(delta_ticks),
                # Aliases (Phase 3.3a prompt) for easier log/backtest parsing
                "activate_ticks": int(self.state.swing_trail_activate_ticks),
                "peak_ticks": int(self.state.swing_trail_peak_ticks),
                "delta_ticks": int(delta_ticks),
                "max_hold_ms": int(max_hold_ms),
                "hold_ms": int(hold_ms),
                "entry_cooldown_left_ms": max(0, int(self.state.swing_entry_cooldown_until_ms) - int(now_ms)),
                "qty_calc": 0.0,
                "qty_used": float(abs(pos)),
                "risk_usdt": 0.0,
                "max_notional": float(max_notional),
                "max_notional_pct": float(getattr(self.s, "STRATEGY_MAX_NOTIONAL_PCT", 0.10)),
                "max_entry_qty": float((max_notional / mid) if mid > 0 else 0.0),
                "max_qty": float((max_notional / mid) if mid > 0 else 0.0),
                "cap_hit": False,
                "exit_reason": str(exit_reason or ""),
                "no_trade_reason": str(no_trade_reason),
            }
            meta.update(dict(phase15_meta_base))
            try:
                meta["gate_state"] = self._gate_state_from_reason(str(no_trade_reason))
                meta.update(self._analyzer.meta(self._analyzer.explain(meta, analyzer_metrics)))
            except Exception:
                pass
            await self._set_meta(snap, now_ms, meta)
            self._maybe_record_frame(
                now_ms,
                snap,
                pred=pred,
                meta=meta,
                trade_pressure=float(press),
                churn_ps=float(churn_ps),
            )
            self._maybe_status_log(
                now_ms,
                mode="swing_1m",
                symbol=str(getattr(self.s, "SYMBOL", "")),
                bot_running=bot_running,
                kill_switch=kill_switch,
                reason=str(no_trade_reason),
                in_pos=True,
                pos_qty=float(pos),
                avg_px=float(avg),
                open_orders=len(open_orders),
                swing_dir=int(swing_dir),
                swing_strength=float(swing_strength),
                swing_edge_ticks=float(swing_edge_ticks),
                swing_score_ticks=float(swing_score_ticks),
                trade_pressure=float(press),
                fee_ticks=int(fee_ticks),
                min_profit_ticks=int(min_profit_ticks),
                activate_ticks=int(self.state.swing_trail_activate_ticks),
                regime=str(regime_class),
                cooldown_left_ms=0,
                trail_active=bool(self.state.swing_trail_active),
                trail_peak_ticks=int(self.state.swing_trail_peak_ticks),
                pos_delta_ticks=int(delta_ticks),
            )
            return

        # --- Flat: entry logic (one-sided, pullback, EV gate) ---

        # If any stray orders exist (from prior mode), keep logic deterministic: only 0..1 order in swing flat mode.
        if len(open_orders) > 1:
            # cancel the newest extras (avoid loops)
            extras = sorted(open_orders, key=lambda x: int(x.get("ts_ms", 0) or 0), reverse=True)[1:]
            for o in extras:
                oid = str(o.get("order_id") or o.get("id") or "")
                await _cancel_order_id(oid)

        open_orders = [o for o in (snap.get("orders", []) or []) if str(o.get("status", "")) in ("NEW", "PARTIALLY_FILLED")]

        entry_cooldown_left = max(0, int(self.state.swing_entry_cooldown_until_ms) - int(now_ms))

        # Effective (live) knobs from bounded autotuner / safety layer (entry only)
        min_edge_mult_live = float(getattr(live_params, "min_edge_mult_live", self._swing_min_edge_mult))
        entry_cd_ms_live = int(getattr(live_params, "entry_cooldown_ms", self._swing_entry_cd_ms))
        rq_min_ms_live = int(getattr(live_params, "requote_min_ms", self._swing_rq_min_ms))
        offset_adj_ticks = int(getattr(live_params, "offset_adj_ticks", 0))
        trading_enabled = bool(getattr(live_params, "trading_enabled", True))

        # Safety pause: NEW entries disabled, reduce-only exits always allowed (handled above)
        if not trading_enabled:
            base_reason = "PAUSED" if not str(getattr(live_params, "pause_reason", "")) else f"PAUSED:{str(getattr(live_params, 'pause_reason', ''))}"
            no_trade_reason = base_reason

            # If we have a resting entry order while paused, cancel it (throttled)
            if open_orders:
                o = open_orders[0]
                oid = str(o.get("order_id") or o.get("id") or "")
                o_ts = int(o.get("ts_ms", 0) or 0)
                age_ms = int(now_ms) - int(o_ts)
                if age_ms < int(self._swing_min_order_life_ms):
                    no_trade_reason = f"{base_reason}:ORDER_YOUNG"
                elif (int(now_ms) - int(getattr(self.state, "pause_last_cancel_ms", 0) or 0)) < 5_000:
                    no_trade_reason = f"{base_reason}:CANCEL_THROTTLE"
                else:
                    await _cancel_order_id(oid)
                    self.state.pause_last_cancel_ms = int(now_ms)
                    no_trade_reason = f"{base_reason}:CANCELLED"

            equity = float(acct.get("equity_usdt", 0.0) or 0.0)
            max_notional = max(0.0, float(equity) * float(getattr(self.s, "STRATEGY_MAX_NOTIONAL_PCT", 0.10)))

            meta = {
                "mode": "swing_1m",
                "symbol": str(getattr(self.s, "SYMBOL", "")),
                "no_trade_reason": str(no_trade_reason),
                "regime_state": str(regime_class),
                "fee_ticks": int(fee_ticks),
                "min_profit_ticks": int(min_profit_ticks),
                "gate_ticks": int(math.ceil(float(min_profit_ticks) * float(min_edge_mult_effective))),
                "swing_dir": int(swing_dir),
                "swing_strength": float(swing_strength),
                "swing_edge_ticks": float(swing_edge_ticks),
                "swing_score_ticks": float(swing_score_ticks),
                "atr_ticks_120": float(atr_ticks_120),
                "slope_norm": float(slope_norm),
                "imb_ema": float(imb_ema),
                "imb_std": float(imb_std_sig),
                "mark_last_div_ticks": float(mark_last_div_ticks),
                "trade_pressure": float(press),
                "entry_offset_ticks": 0,
                "entry_price": 0.0,
                "tp_ticks": int(tp_ticks),
                "sl_ticks": int(sl_ticks),
                "max_hold_ms": int(max_hold_ms),
                "hold_ms": 0,
                "entry_cooldown_left_ms": int(entry_cooldown_left),
                "qty_calc": 0.0,
                "qty_used": 0.0,
                "risk_usdt": 0.0,
                "max_notional": float(max_notional),
                "max_qty": float((max_notional / mid) if mid > 0 else 0.0),
                "cap_hit": False,
                "exit_reason": "",
            }
            meta.update(dict(phase15_meta_base))
            try:
                meta["gate_state"] = self._gate_state_from_reason(str(no_trade_reason))
                meta.update(self._analyzer.meta(self._analyzer.explain(meta, analyzer_metrics)))
            except Exception:
                pass

            await self._set_meta(snap, now_ms, meta)
            self._maybe_record_frame(
                now_ms,
                snap,
                pred=pred,
                meta=meta,
                trade_pressure=float(press),
                churn_ps=float(churn_ps),
            )
            self._maybe_status_log(
                now_ms,
                mode="swing_1m",
                symbol=str(getattr(self.s, "SYMBOL", "")),
                bot_running=bot_running,
                kill_switch=kill_switch,
                reason=str(no_trade_reason),
                in_pos=False,
                pos_qty=float(pos),
                avg_px=float(avg),
                open_orders=len(open_orders),
                swing_dir=int(swing_dir),
                swing_strength=float(swing_strength),
                swing_edge_ticks=float(swing_edge_ticks),
                swing_score_ticks=float(swing_score_ticks),
                trade_pressure=float(press),
                fee_ticks=int(fee_ticks),
                min_profit_ticks=int(min_profit_ticks),
                regime=str(regime_class),
                cooldown_left_ms=int(entry_cooldown_left),
            )
            return

        # EV gate & regime gate
        gate_ok = (
            swing_dir != 0
            and (not bool(toxic_hard))
            and float(swing_edge_ticks) >= float(min_profit_ticks) * float(min_edge_mult_effective)
        )

        desired_side = "BUY" if swing_dir > 0 else ("SELL" if swing_dir < 0 else "")
        entry_offset_ticks = 0
        entry_px = 0.0

        # Apply gate-hold: if gate flips off, do not cancel very fresh orders immediately
        if not gate_ok:
            # Phase 0: explain why we're idle
            no_trade_reason = "GATED"
            if swing_dir == 0:
                no_trade_reason = "NO_SIGNAL"
            elif bool(toxic_hard):
                no_trade_reason = "TOXIC_HARD"
            elif bool(toxic_soft):
                no_trade_reason = "TOXIC_SOFT"
            elif float(swing_edge_ticks) < float(min_profit_ticks) * float(min_edge_mult_effective):
                no_trade_reason = "EV_GATE"

            if open_orders:
                o = open_orders[0]
                o_ts = int(o.get("ts_ms", 0) or 0)
                age_ms = int(now_ms) - int(o_ts)
                if age_ms >= int(self._swing_gate_hold_ms):
                    oid = str(o.get("order_id") or o.get("id") or "")
                    await _cancel_order_id(oid)
                    no_trade_reason = f"{no_trade_reason}:CANCELLED"
                else:
                    no_trade_reason = f"{no_trade_reason}:ORDER_YOUNG"
            self.state.swing_gate_off_since_ms = int(self.state.swing_gate_off_since_ms or now_ms)

            # Meta while flat + gated
            equity = float(acct.get("equity_usdt", 0.0) or 0.0)
            max_notional = max(0.0, float(equity) * float(getattr(self.s, "STRATEGY_MAX_NOTIONAL_PCT", 0.10)))
            meta = {
                "mode": "swing_1m",
                "symbol": str(getattr(self.s, "SYMBOL", "")),
                "no_trade_reason": str(no_trade_reason),
                "regime_state": str(regime_class),
                "size_mult": 1.0,
                "base_seed_qty": 0.0,
                "base_notional_pct": 0.0,
                "side_mult_buy": 1.0,
                "side_mult_sell": 1.0,
                "dd_now": 0.0,
                "side_gate_buy": 1.0,
                "side_gate_sell": 1.0,
                "side_inside_adj_buy": 0,
                "side_inside_adj_sell": 0,
                "side_pnl_avg_buy": 0.0,
                "side_pnl_avg_sell": 0.0,
                "side_trades_buy": 0,
                "side_trades_sell": 0,
                "side_fq_n": 0,
                "fee_ticks": int(fee_ticks),
                "min_profit_ticks": int(min_profit_ticks),
                "min_inside_ticks": int(min_profit_ticks),
                "gate_ticks": int(math.ceil(float(min_profit_ticks) * float(min_edge_mult_effective))),
                "inside_used": 0,
                "swing_dir": int(swing_dir),
                "swing_strength": float(swing_strength),
                "swing_edge_ticks": float(swing_edge_ticks),
                "swing_score_ticks": float(swing_score_ticks),
                "atr_ticks_120": float(atr_ticks_120),
                "slope_norm": float(slope_norm),
                "imb_ema": float(imb_ema),
                "imb_std": float(imb_std_sig),
                "mark_last_div_ticks": float(mark_last_div_ticks),
                "trade_pressure": float(press),
                "ml_p_up": 0.5,
                "ml_edge": 0.0,
                "ml_h": 0,
                "ml_n": 0,
                "bandit_arm": "",
                "bandit_eps": 0.0,
                "entry_offset_ticks": 0,
                "entry_price": 0.0,
                "tp_ticks": int(tp_ticks),
                "sl_ticks": int(sl_ticks),
                "max_hold_ms": int(max_hold_ms),
                "hold_ms": 0,
                "entry_cooldown_left_ms": int(entry_cooldown_left),
                "qty_calc": 0.0,
                "qty_used": 0.0,
                "risk_usdt": 0.0,
                "max_notional": float(max_notional),
                "max_notional_pct": float(getattr(self.s, "STRATEGY_MAX_NOTIONAL_PCT", 0.10)),
                "max_entry_qty": float((max_notional / mid) if mid > 0 else 0.0),
                "max_qty": float((max_notional / mid) if mid > 0 else 0.0),
                "cap_hit": False,
                "exit_reason": "",
                "no_trade_reason": str(no_trade_reason),
            }
            meta.update(dict(phase15_meta_base))
            try:
                meta["gate_state"] = self._gate_state_from_reason(str(no_trade_reason))
                meta.update(self._analyzer.meta(self._analyzer.explain(meta, analyzer_metrics)))
            except Exception:
                pass
            await self._set_meta(snap, now_ms, meta)
            self._maybe_record_frame(
                now_ms,
                snap,
                pred=pred,
                meta=meta,
                trade_pressure=float(press),
                churn_ps=float(churn_ps),
            )
            self._maybe_status_log(
                now_ms,
                mode="swing_1m",
                symbol=str(getattr(self.s, "SYMBOL", "")),
                bot_running=bot_running,
                kill_switch=kill_switch,
                reason=str(no_trade_reason),
                in_pos=False,
                pos_qty=float(pos),
                avg_px=float(avg),
                open_orders=len(open_orders),
                swing_dir=int(swing_dir),
                swing_strength=float(swing_strength),
                swing_edge_ticks=float(swing_edge_ticks),
                swing_score_ticks=float(swing_score_ticks),
                trade_pressure=float(press),
                fee_ticks=int(fee_ticks),
                min_profit_ticks=int(min_profit_ticks),
                regime=str(regime_class),
                cooldown_left_ms=int(entry_cooldown_left),
            )
            return

        # Gate is ON
        self.state.swing_gate_off_since_ms = 0

        # entry price offset (fees+vol aware pullback)
        entry_offset_ticks = max(
            int(self._swing_entry_base_off),
            int(min_profit_ticks * float(self._swing_entry_off_fees)),
            int(vol_ticks * float(self._swing_entry_off_vol)),
        )

        # Autotuner may add/subtract a small bounded adjustment (ENTRY only)
        try:
            entry_offset_ticks = max(0, int(entry_offset_ticks) + int(offset_adj_ticks) + int(regime_offset_add_ticks))
        except Exception:
            entry_offset_ticks = int(entry_offset_ticks)

        if tick > 0 and mid > 0 and desired_side:
            raw_px = (mid - entry_offset_ticks * tick) if desired_side == "BUY" else (mid + entry_offset_ticks * tick)

            # Align to tick grid safely
            if desired_side == "BUY":
                # floor to tick
                entry_px = math.floor(raw_px / tick) * tick
                # maker-only clamp
                if ba > 0:
                    entry_px = min(entry_px, ba - max(tick * 0.25, 1e-9))
            else:
                # ceil to tick
                entry_px = math.ceil(raw_px / tick) * tick
                if bb > 0:
                    entry_px = max(entry_px, bb + max(tick * 0.25, 1e-9))

        # sizing (risk-based with notional cap)
        equity = float(acct.get("equity_usdt", 0.0) or 0.0)
        max_notional = max(0.0, float(equity) * float(getattr(self.s, "STRATEGY_MAX_NOTIONAL_PCT", 0.10)))
        max_qty = (max_notional / mid) if mid > 0 else 0.0

        risk_usdt = max(0.0, float(equity) * float(self._swing_risk_pct))
        qty_calc = (risk_usdt / (float(sl_ticks) * float(tick))) if (sl_ticks > 0 and tick > 0) else 0.0
        qty_used = max(0.0, min(float(qty_calc), float(max_qty)))
        cap_hit = bool(qty_calc > max_qty + 1e-12)

        # If in cooldown, do not place/requote; just keep resting order if any
        if entry_cooldown_left > 0:
            no_trade_reason = "COOLDOWN" if not open_orders else "COOLDOWN:ORDER_RESTING"
            meta = {
                "mode": "swing_1m",
                "symbol": str(getattr(self.s, "SYMBOL", "")),
                "no_trade_reason": str(no_trade_reason),
                "regime_state": str(regime_class),
                "size_mult": 1.0,
                "base_seed_qty": float(qty_used),
                "base_notional_pct": 0.0,
                "side_mult_buy": 1.0,
                "side_mult_sell": 1.0,
                "dd_now": 0.0,
                "side_gate_buy": 1.0,
                "side_gate_sell": 1.0,
                "side_inside_adj_buy": 0,
                "side_inside_adj_sell": 0,
                "side_pnl_avg_buy": 0.0,
                "side_pnl_avg_sell": 0.0,
                "side_trades_buy": 0,
                "side_trades_sell": 0,
                "side_fq_n": 0,
                "fee_ticks": int(fee_ticks),
                "min_profit_ticks": int(min_profit_ticks),
                "min_inside_ticks": int(min_profit_ticks),
                "gate_ticks": int(math.ceil(float(min_profit_ticks) * float(min_edge_mult_effective))),
                "inside_used": int(entry_offset_ticks),
                "swing_dir": int(swing_dir),
                "swing_strength": float(swing_strength),
                "swing_edge_ticks": float(swing_edge_ticks),
                "swing_score_ticks": float(swing_score_ticks),
                "atr_ticks_120": float(atr_ticks_120),
                "slope_norm": float(slope_norm),
                "imb_ema": float(imb_ema),
                "imb_std": float(imb_std_sig),
                "mark_last_div_ticks": float(mark_last_div_ticks),
                "trade_pressure": float(press),
                "ml_p_up": 0.5,
                "ml_edge": 0.0,
                "ml_h": 0,
                "ml_n": 0,
                "bandit_arm": "",
                "bandit_eps": 0.0,
                "entry_offset_ticks": int(entry_offset_ticks),
                "entry_price": float(entry_px),
                "tp_ticks": int(tp_ticks),
                "sl_ticks": int(sl_ticks),
                "max_hold_ms": int(max_hold_ms),
                "hold_ms": 0,
                "entry_cooldown_left_ms": int(entry_cooldown_left),
                "qty_calc": float(qty_calc),
                "qty_used": float(qty_used),
                "risk_usdt": float(risk_usdt),
                "max_notional": float(max_notional),
                "max_notional_pct": float(getattr(self.s, "STRATEGY_MAX_NOTIONAL_PCT", 0.10)),
                "max_entry_qty": float(max_qty),
                "max_qty": float(max_qty),
                "cap_hit": bool(cap_hit),
                "exit_reason": "",
                "no_trade_reason": str(no_trade_reason),
            }
            meta.update(dict(phase15_meta_base))
            try:
                meta["gate_state"] = self._gate_state_from_reason(str(no_trade_reason))
                meta.update(self._analyzer.meta(self._analyzer.explain(meta, analyzer_metrics)))
            except Exception:
                pass
            await self._set_meta(snap, now_ms, meta)
            self._maybe_record_frame(
                now_ms,
                snap,
                pred=pred,
                meta=meta,
                trade_pressure=float(press),
                churn_ps=float(churn_ps),
            )
            self._maybe_status_log(
                now_ms,
                mode="swing_1m",
                symbol=str(getattr(self.s, "SYMBOL", "")),
                bot_running=bot_running,
                kill_switch=kill_switch,
                reason=str(no_trade_reason),
                in_pos=False,
                pos_qty=float(pos),
                avg_px=float(avg),
                open_orders=len(open_orders),
                swing_dir=int(swing_dir),
                swing_strength=float(swing_strength),
                swing_edge_ticks=float(swing_edge_ticks),
                swing_score_ticks=float(swing_score_ticks),
                trade_pressure=float(press),
                fee_ticks=int(fee_ticks),
                min_profit_ticks=int(min_profit_ticks),
                regime=str(regime_class),
                cooldown_left_ms=int(entry_cooldown_left),
            )
            return

        no_trade_reason = "ENTRY_IDLE"

        # Manage/ensure a single entry order on desired side
        existing = open_orders[0] if open_orders else None
        if existing and str(existing.get("side", "")) != desired_side:
            # cancel if not too fresh (avoid flip-flop)
            oid = str(existing.get("order_id") or existing.get("id") or "")
            age_ms = int(now_ms) - int(existing.get("ts_ms", 0) or 0)
            if age_ms >= int(self._swing_min_order_life_ms):
                await _cancel_order_id(oid)
                no_trade_reason = "FLIP_CANCELLED"
            existing = None
            if age_ms < int(self._swing_min_order_life_ms):
                no_trade_reason = "FLIP_WAIT:ORDER_YOUNG"

        if not existing:
            if not desired_side:
                no_trade_reason = "NO_DESIRED_SIDE"
            elif entry_px <= 0:
                no_trade_reason = "BAD_ENTRY_PRICE"
            elif qty_used <= 0:
                no_trade_reason = "QTY_ZERO"
            else:
                o = await self.engine.place_limit_post_only(desired_side, price=float(entry_px), qty=float(qty_used))
                self.state.swing_last_entry_attempt_ms = int(now_ms)
                self.state.swing_entry_cooldown_until_ms = int(now_ms) + int(entry_cd_ms_live)
                if str(getattr(o, "status", "")) == "REJECTED":
                    no_trade_reason = "ENTRY_REJECTED"
                else:
                    no_trade_reason = "ENTRY_PLACED"
                    # rolling execution: count accepted attempts + start time-to-fill timer
                    self.state.entry_attempt_ts.append(int(now_ms))
                    self.state.pending_entry_ts_ms = int(now_ms)
        else:
            # Requote only if drift is large AND order is not too fresh AND min interval satisfied
            oid = str(existing.get("order_id") or existing.get("id") or "")
            o_ts = int(existing.get("ts_ms", 0) or 0)
            age_ms = int(now_ms) - int(o_ts)
            o_px = float(existing.get("price", 0.0) or 0.0)
            drift_ticks = abs(float(entry_px) - float(o_px)) / float(tick) if tick > 0 else 0.0
            can_rq = (int(now_ms) - int(self.state.swing_last_requote_ms or 0)) >= int(rq_min_ms_live)
            # Default reason while order is resting
            if age_ms < int(self._swing_min_order_life_ms):
                no_trade_reason = "ORDER_YOUNG"
            elif drift_ticks < float(self._swing_rq_drift_ticks):
                no_trade_reason = "DRIFT_SMALL"
            elif not can_rq:
                no_trade_reason = "REQUOTE_THROTTLE"
            else:
                no_trade_reason = "REQUOTE_ELIGIBLE"

            if (
                age_ms >= int(self._swing_min_order_life_ms)
                and drift_ticks >= float(self._swing_rq_drift_ticks)
                and can_rq
                and entry_px > 0
                and qty_used > 0
                and oid
            ):
                await _cancel_order_id(oid)
                self.state.swing_last_requote_ms = int(now_ms)
                o2 = await self.engine.place_limit_post_only(desired_side, price=float(entry_px), qty=float(qty_used))
                self.state.swing_last_entry_attempt_ms = int(now_ms)
                self.state.swing_entry_cooldown_until_ms = int(now_ms) + int(entry_cd_ms_live)
                if str(getattr(o2, "status", "")) == "REJECTED":
                    no_trade_reason = "REQUOTE_REJECTED"
                else:
                    no_trade_reason = "REQUOTE_PLACED"
                    # rolling execution: count accepted attempts + restart time-to-fill timer
                    self.state.entry_attempt_ts.append(int(now_ms))
                    self.state.pending_entry_ts_ms = int(now_ms)

        # Meta while flat
        meta = {
            "mode": "swing_1m",
            "symbol": str(getattr(self.s, "SYMBOL", "")),
            "no_trade_reason": str(no_trade_reason),
            "regime_state": str(regime_class),
                "size_mult": 1.0,
                "base_seed_qty": float(qty_used),
                "base_notional_pct": 0.0,
                "side_mult_buy": 1.0,
                "side_mult_sell": 1.0,
                "dd_now": 0.0,
                "side_gate_buy": 1.0,
                "side_gate_sell": 1.0,
                "side_inside_adj_buy": 0,
                "side_inside_adj_sell": 0,
                "side_pnl_avg_buy": 0.0,
                "side_pnl_avg_sell": 0.0,
                "side_trades_buy": 0,
                "side_trades_sell": 0,
                "side_fq_n": 0,
            "fee_ticks": int(fee_ticks),
            "min_profit_ticks": int(min_profit_ticks),
                "min_inside_ticks": int(min_profit_ticks),
                "gate_ticks": int(math.ceil(float(min_profit_ticks) * float(min_edge_mult_effective))),
                "inside_used": int(entry_offset_ticks),
            "swing_dir": int(swing_dir),
            "swing_strength": float(swing_strength),
            "swing_edge_ticks": float(swing_edge_ticks),
            "swing_score_ticks": float(swing_score_ticks),
            "atr_ticks_120": float(atr_ticks_120),
            "slope_norm": float(slope_norm),
            "imb_ema": float(imb_ema),
            "imb_std": float(imb_std_sig),
            "mark_last_div_ticks": float(mark_last_div_ticks),
            "trade_pressure": float(press),
                "ml_p_up": 0.5,
                "ml_edge": 0.0,
                "ml_h": 0,
                "ml_n": 0,
                "bandit_arm": "",
                "bandit_eps": 0.0,
            "entry_offset_ticks": int(entry_offset_ticks),
            "entry_price": float(entry_px),
            "tp_ticks": int(tp_ticks),
            "sl_ticks": int(sl_ticks),
            "max_hold_ms": int(max_hold_ms),
            "hold_ms": 0,
            "entry_cooldown_left_ms": max(0, int(self.state.swing_entry_cooldown_until_ms) - int(now_ms)),
            "qty_calc": float(qty_calc),
            "qty_used": float(qty_used),
            "risk_usdt": float(risk_usdt),
            "max_notional": float(max_notional),
                "max_notional_pct": float(getattr(self.s, "STRATEGY_MAX_NOTIONAL_PCT", 0.10)),
                "max_entry_qty": float(max_qty),
            "max_qty": float(max_qty),
            "cap_hit": bool(cap_hit),
            "exit_reason": "",
            "no_trade_reason": str(no_trade_reason),
        }
        meta.update(dict(phase15_meta_base))
        try:
            meta["gate_state"] = self._gate_state_from_reason(str(no_trade_reason))
            meta.update(self._analyzer.meta(self._analyzer.explain(meta, analyzer_metrics)))
        except Exception:
            pass
        await self._set_meta(snap, now_ms, meta)

        self._maybe_record_frame(
            now_ms,
            snap,
            pred=pred,
            meta=meta,
            trade_pressure=float(press),
            churn_ps=float(churn_ps),
        )

        self._maybe_status_log(
            now_ms,
            mode="swing_1m",
            symbol=str(getattr(self.s, "SYMBOL", "")),
            bot_running=bot_running,
            kill_switch=kill_switch,
            reason=str(no_trade_reason),
            in_pos=False,
            pos_qty=float(pos),
            avg_px=float(avg),
            open_orders=len(open_orders),
            swing_dir=int(swing_dir),
            swing_strength=float(swing_strength),
            swing_edge_ticks=float(swing_edge_ticks),
            swing_score_ticks=float(swing_score_ticks),
            trade_pressure=float(press),
            fee_ticks=int(fee_ticks),
            min_profit_ticks=int(min_profit_ticks),
            regime=str(regime_for_autotune),
            cooldown_left_ms=max(0, int(self.state.swing_entry_cooldown_until_ms) - int(now_ms)),
        )

    def _apply_level1_throttles(self, snap: dict, now_ms: int) -> tuple[float, int, float]:
        """Level 1 risk throttles (simple, low-risk).

        Returns:
          qty_mult: multiplier for ENTRY order qty (exit remains full size)
          quote_interval_ms: minimum time between ENTRY order placements
          entry_mult: multiplier for entry threshold (higher => fewer entries)

        Also updates loss_streak and may extend cooldown_until_ms.
        """
        st = snap.get("stats", {}) or {}
        trades_n = int(st.get("trades_n", 0) or 0)

        # Reset on session reset (trades counter dropped)
        if trades_n < int(self.state.trades_seen):
            self.state.trades_seen = trades_n
            self.state.loss_streak = 0

        # Process newly closed trades (net < 0 increments streak; win resets streak)
        delta = trades_n - int(self.state.trades_seen)
        if delta > 0:
            # closed_trades in snapshot are newest-first (appendleft in store)
            recent = list(snap.get("closed_trades", []) or [])[:delta]
            for ct in reversed(recent):  # oldest -> newest
                net = float(ct.get("realized_pnl_usdt", 0.0) or 0.0) - float(ct.get("fees_usdt", 0.0) or 0.0)
                # Phase-2: side performance tracking (attribute trade to entry-side)
                # ClosedTrade.side is LONG/SHORT -> BUY/SELL entry side.
                raw = str(ct.get("side", "") or "").upper()
                side = "BUY" if raw == "LONG" else ("SELL" if raw == "SHORT" else "")
                ts = int(ct.get("exit_ts_ms", 0) or ct.get("ts_ms", 0) or now_ms)
                if side:
                    self.state.side_perf_hist.append((ts, side, float(net)))
                if net < 0:
                    self.state.loss_streak += 1
                else:
                    self.state.loss_streak = 0

                # Dynamic cooldown after N consecutive losing CLOSED trades
                if int(self.state.loss_streak) >= int(self._dyn_cd_streak):
                    k = int(self.state.loss_streak) - int(self._dyn_cd_streak)
                    cd = int(self._dyn_cd_min_ms * (2 ** k))
                    cd = min(int(self._dyn_cd_max_ms), max(int(self._dyn_cd_min_ms), cd))
                    self.state.cooldown_until_ms = max(int(self.state.cooldown_until_ms), int(now_ms) + int(cd))
                    log.info("dyn_cooldown loss_streak=%d cooldown_ms=%d", int(self.state.loss_streak), int(cd))

            self.state.trades_seen = trades_n

        # Soft-stop: reduce aggressiveness while in mild drawdown regime
        if int(self.state.loss_streak) >= int(self._softstop_streak):
            return float(self._softstop_qty_mult), int(self._softstop_quote_interval_ms), float(self._softstop_entry_mult)

        return 1.0, 0, 1.0


    def _trim_by_cutoff_pair(self, dq: Deque[Tuple[int, float]], cutoff_ms: int) -> None:
        while dq and int(dq[0][0]) < int(cutoff_ms):
            dq.popleft()

    def _trim_by_cutoff_int(self, dq: Deque[int], cutoff_ms: int) -> None:
        while dq and int(dq[0]) < int(cutoff_ms):
            dq.popleft()

    def _std(self, vals: list[float]) -> float:
        if len(vals) < 2:
            return 0.0
        mean = sum(vals) / len(vals)
        var = sum((x - mean) ** 2 for x in vals) / max(len(vals) - 1, 1)
        return float(math.sqrt(var))

    def _update_regime(self, now_ms: int, bb: float, ba: float, tick: float, mid: float, imb: float) -> dict:
        """Compute simple regime metrics over a rolling window.

        Metrics:
          - churn: bb/ba changes per second (within win)
          - imbalance stability: std(imbalance) (within win)
          - trend: mid change over win (in ticks)

        Notes:
          - We keep a longer baseline history (Phase 3.1) so we can compute
            v2 toxic scores (spread/vol spikes vs median) without losing history.
        """
        now_ms = int(now_ms)
        win = int(self._regime_window_ms)
        baseline_ms = int(getattr(self.s, "STRATEGY_REGIME_V2_BASELINE_MS", 60_000) or 60_000)
        keep_ms = max(int(win), int(baseline_ms))

        cutoff_keep = now_ms - keep_ms
        cutoff_win = now_ms - win

        # keep longer history for v2
        self.state.regime_mid_hist.append((now_ms, float(mid)))
        self.state.regime_imb_hist.append((now_ms, float(imb)))
        self._trim_by_cutoff_pair(self.state.regime_mid_hist, cutoff_keep)
        self._trim_by_cutoff_pair(self.state.regime_imb_hist, cutoff_keep)

        # churn (best bid/ask changes)
        if float(tick) > 0:
            bb_tick = int(round(float(bb) / float(tick)))
            ba_tick = int(round(float(ba) / float(tick)))
        else:
            bb_tick = int(round(float(bb) * 10_000))
            ba_tick = int(round(float(ba) * 10_000))

        if self.state.regime_last_bb_tick == 0 and self.state.regime_last_ba_tick == 0:
            self.state.regime_last_bb_tick = bb_tick
            self.state.regime_last_ba_tick = ba_tick
        else:
            if bb_tick != int(self.state.regime_last_bb_tick) or ba_tick != int(self.state.regime_last_ba_tick):
                self.state.regime_churn_ts.append(now_ms)
                self.state.regime_last_bb_tick = bb_tick
                self.state.regime_last_ba_tick = ba_tick

        # churn window stays at win
        self._trim_by_cutoff_int(self.state.regime_churn_ts, cutoff_win)
        churn_ps = float(len(self.state.regime_churn_ts)) / max(float(win) / 1000.0, 1e-9)

        # imbalance std within win
        imb_vals = [v for ts, v in self.state.regime_imb_hist if int(ts) >= int(cutoff_win)]
        imb_std = self._std(imb_vals)

        # trend within win
        trend_ticks = 0.0
        if float(tick) > 0:
            mids = [(ts, px) for ts, px in self.state.regime_mid_hist if int(ts) >= int(cutoff_win)]
            if len(mids) >= 2:
                first_mid = float(mids[0][1])
                last_mid = float(mids[-1][1])
                trend_ticks = (last_mid - first_mid) / float(tick)

        # log (no spam) every ~5s
        if (now_ms - int(self.state.regime_last_log_ms)) >= 5_000:
            self.state.regime_last_log_ms = now_ms
            log.info("regime win_ms=%d churn_ps=%.2f imb_std=%.3f trend_ticks=%.2f", win, churn_ps, imb_std, trend_ticks)

        return {
            "churn_ps": churn_ps,
            "imb_std": imb_std,
            "trend_ticks": trend_ticks,
        }

    def _classify_regime_v2(
        self,
        now_ms: int,
        *,
        tick: float,
        mid: float,
        spread_ticks: float,
        vol_ticks: float,
        churn_ps: float,
    ) -> dict:
        """Phase 3.1: classify market regime (trend/chop/toxic) with soft guards.

        - trend/chop via efficiency ratio (net move / path length) over a time window
        - toxic via spread/vol spikes vs rolling median + churn spikes

        Returns dict with scores + state.
        """
        now_ms = int(now_ms)
        tick = float(tick)
        mid = float(mid)
        sp = max(0.0, float(spread_ticks))
        vt = max(0.0, float(vol_ticks))
        churn = max(0.0, float(churn_ps))

        win_ms = int(getattr(self.s, "STRATEGY_REGIME_V2_WINDOW_MS", 15_000) or 15_000)
        baseline_ms = int(getattr(self.s, "STRATEGY_REGIME_V2_BASELINE_MS", 60_000) or 60_000)
        cutoff_keep = now_ms - baseline_ms
        cutoff_win = now_ms - win_ms

        # keep baseline samples
        self.state.regime_spread_hist.append((now_ms, float(sp)))
        self.state.regime_vol_ticks_hist.append((now_ms, float(vt)))
        self._trim_by_cutoff_pair(self.state.regime_spread_hist, cutoff_keep)
        self._trim_by_cutoff_pair(self.state.regime_vol_ticks_hist, cutoff_keep)

        def _median(vals):
            if not vals:
                return 0.0
            arr = sorted(float(x) for x in vals)
            return float(arr[len(arr) // 2])

        spread_med = _median([v for ts, v in self.state.regime_spread_hist if int(ts) >= int(cutoff_keep)])
        vol_med = _median([v for ts, v in self.state.regime_vol_ticks_hist if int(ts) >= int(cutoff_keep)])

        # guard against tiny baselines
        spread_med = max(1.0, float(spread_med))
        vol_med = max(1e-6, float(vol_med))

        # trend/chop via efficiency ratio
        trend_ticks = 0.0
        path_ticks = 0.0
        eff_ratio = 0.0
        if tick > 0:
            mids = [(ts, px) for ts, px in self.state.regime_mid_hist if int(ts) >= int(cutoff_win)]
            if len(mids) >= 2:
                first = float(mids[0][1])
                last = float(mids[-1][1])
                trend_ticks = (last - first) / float(tick)
                prev = first
                for _, px in mids[1:]:
                    path_ticks += abs(float(px) - float(prev)) / float(tick)
                    prev = float(px)
                if path_ticks > 0:
                    eff_ratio = abs(float(trend_ticks)) / max(float(path_ticks), 1e-9)
                    eff_ratio = max(0.0, min(1.0, float(eff_ratio)))

        trend_strength = float(eff_ratio)
        chop_score = float(max(0.0, min(1.0, 1.0 - float(eff_ratio))))

        # toxic score: normalized spikes vs baseline medians
        churn_ref = max(1e-6, float(getattr(self.s, "STRATEGY_REGIME_MAX_CHURN_PER_S", 8.0) or 8.0))
        spread_mult = float(sp) / max(float(spread_med), 1e-6)
        vol_mult = float(vt) / max(float(vol_med), 1e-6)
        churn_mult = float(churn) / float(churn_ref)

        sp_soft = max(1e-6, float(getattr(self.s, "STRATEGY_REGIME_V2_SPREAD_SPIKE_SOFT", getattr(self.s, "STRATEGY_REGIME_V2_SPREAD_SPIKE_MULT_SOFT", 1.6)) or 1.6))
        vol_soft = max(1e-6, float(getattr(self.s, "STRATEGY_REGIME_V2_VOL_SPIKE_SOFT", getattr(self.s, "STRATEGY_REGIME_V2_VOL_SPIKE_MULT_SOFT", 1.6)) or 1.6))
        churn_soft = max(1e-6, float(getattr(self.s, "STRATEGY_REGIME_V2_CHURN_SPIKE_SOFT", getattr(self.s, "STRATEGY_REGIME_V2_CHURN_SPIKE_MULT_SOFT", 1.5)) or 1.5))

        tox_s = float(spread_mult) / float(sp_soft)
        tox_v = float(vol_mult) / float(vol_soft)
        tox_c = float(churn_mult) / float(churn_soft)
        toxic_score = float(max(tox_s, tox_v, tox_c))

        tox_soft_thr = float(getattr(self.s, "STRATEGY_REGIME_V2_TOXIC_SOFT_SCORE", getattr(self.s, "STRATEGY_REGIME_V2_TOXIC_SOFT_THR", 1.0)) or 1.0)
        tox_hard_thr = float(getattr(self.s, "STRATEGY_REGIME_V2_TOXIC_HARD_SCORE", getattr(self.s, "STRATEGY_REGIME_V2_TOXIC_HARD_THR", 1.4)) or 1.4)

        trend_er_thr = float(getattr(self.s, "STRATEGY_REGIME_V2_TREND_ER_THR", 0.35) or 0.35)
        trend_ticks_min = float(getattr(self.s, "STRATEGY_REGIME_V2_TREND_TICKS_MIN", 4.0) or 4.0)
        chop_thr = float(getattr(self.s, "STRATEGY_REGIME_V2_CHOP_THR", getattr(self.s, "STRATEGY_REGIME_V2_CHOP_SCORE_THR", 0.65)) or 0.65)

        toxic_hard = bool(toxic_score >= tox_hard_thr)
        toxic_soft = bool((toxic_score >= tox_soft_thr) and (not toxic_hard))

        is_trend = bool((trend_strength >= trend_er_thr) and (abs(float(trend_ticks)) >= float(trend_ticks_min)))

        state = "NORMAL"
        if toxic_hard:
            state = "TOXIC"
        elif toxic_soft:
            state = "TOXIC_SOFT"
        elif is_trend:
            state = "TREND_UP" if float(trend_ticks) > 0 else "TREND_DOWN"
        elif chop_score >= chop_thr:
            state = "CHOP"

        return {
            "state": str(state),
            "trend_strength": float(trend_strength),
            "chop_score": float(chop_score),
            "toxic_score": float(toxic_score),
            "eff_ratio": float(eff_ratio),
            "trend_ticks": float(trend_ticks),
            "path_ticks": float(path_ticks),
            "spread_med": float(spread_med),
            "vol_med": float(vol_med),
            "spread_mult": float(spread_mult),
            "vol_mult": float(vol_mult),
            "churn_mult": float(churn_mult),
            "toxic_hard": bool(toxic_hard),
            "toxic_soft": bool(toxic_soft),
            "is_trend": bool(is_trend),
        }

    def _fillq_side_mods(self, snap: dict) -> dict:
        """Level-2 fill-quality feedback loop.

        Uses Store stats keys:
          fq_{H}_n, fq_{H}_mean_ticks_buy, fq_{H}_mean_ticks_sell

        If mean outcome after H ms is negative for a side, we:
          - raise the entry threshold for that side
          - add a side-specific quote interval (rate limit)
        """
        st = snap.get("stats", {}) or {}

        # choose horizon: prefer configured H, fall back to 1000ms if configured has too few samples
        h = int(self._fq_horizon_ms)
        n = int(st.get(f"fq_{h}_n", 0) or 0)
        if n < int(self._fq_min_n) and h != 1000:
            h2 = 1000
            n2 = int(st.get(f"fq_{h2}_n", 0) or 0)
            if n2 >= int(self._fq_min_n):
                h = h2
                n = n2

        buy_mean = float(st.get(f"fq_{h}_mean_ticks_buy", 0.0) or 0.0)
        sell_mean = float(st.get(f"fq_{h}_mean_ticks_sell", 0.0) or 0.0)

        # If the chosen horizon is too "flat" (means ~= 0) but 1000ms has signal, use 1000ms.
        if n >= int(self._fq_min_n) and h != 1000:
            if abs(float(buy_mean)) + abs(float(sell_mean)) < 1e-9:
                h2 = 1000
                n2 = int(st.get(f"fq_{h2}_n", 0) or 0)
                if n2 >= int(self._fq_min_n):
                    bm2 = float(st.get(f"fq_{h2}_mean_ticks_buy", 0.0) or 0.0)
                    sm2 = float(st.get(f"fq_{h2}_mean_ticks_sell", 0.0) or 0.0)
                    if abs(bm2) + abs(sm2) > 0:
                        h = h2
                        n = n2
                        buy_mean = bm2
                        sell_mean = sm2

        def penalty(mean_ticks: float) -> float:
            # only penalize negative mean; clamp 0..1
            if mean_ticks >= 0:
                return 0.0
            scale = max(float(self._fq_edge_scale), 1e-9)
            return min(1.0, max(0.0, (-mean_ticks) / scale))

        p_buy = penalty(buy_mean) if n >= int(self._fq_min_n) else 0.0
        p_sell = penalty(sell_mean) if n >= int(self._fq_min_n) else 0.0

        # entry threshold multipliers
        buy_thr_mult = 1.0 + p_buy * (max(float(self._fq_neg_entry_mult_max), 1.0) - 1.0)
        sell_thr_mult = 1.0 + p_sell * (max(float(self._fq_neg_entry_mult_max), 1.0) - 1.0)

        # side quote intervals
        buy_int = int(round(p_buy * float(self._fq_neg_quote_int_max_ms)))
        sell_int = int(round(p_sell * float(self._fq_neg_quote_int_max_ms)))

        return {
            "h": h,
            "n": n,
            "buy_mean": buy_mean,
            "sell_mean": sell_mean,
            "buy_thr_mult": buy_thr_mult,
            "sell_thr_mult": sell_thr_mult,
            "buy_int_ms": buy_int,
            "sell_int_ms": sell_int,
        }

    def _side_perf_mults(self, now_ms: int, fq: Optional[dict] = None, gate_ticks: int = 1) -> tuple[float, float, dict]:
        """Phase 2c: per-side performance control.

        Combines rolling realized net PnL per side with optional FillQ mean ticks
        to compute:
          - buy_mult/sell_mult: scales qty + quote frequency
          - buy_gate_mult/sell_gate_mult: stricter gate for bad side (flat/add-side)
          - buy_inside_adj/sell_inside_adj: widen bad side (reduce inside ticks)
        """
        fq = fq or {}
        cutoff = int(now_ms) - int(self._side_perf_window_ms)
        dq = self.state.side_perf_hist
        while dq and int(dq[0][0]) < cutoff:
            dq.popleft()

        b_net = 0.0; s_net = 0.0; b_n = 0; s_n = 0
        for ts, side, net in dq:
            if side == "BUY":
                b_net += float(net); b_n += 1
            elif side == "SELL":
                s_net += float(net); s_n += 1

        b_avg = b_net / max(1, b_n)
        s_avg = s_net / max(1, s_n)

        target = max(float(self._side_perf_target_usdt), 1e-9)
        min_tr = int(self._side_perf_min_trades)

        def pnl_score(avg: float, n: int) -> float:
            if n < min_tr:
                return 0.0
            x = float(avg) / target
            return max(-1.0, min(1.0, x))

        fq_n = int(fq.get("n", 0) or 0)
        fq_b = float(fq.get("buy_mean", 0.0) or 0.0)
        fq_s = float(fq.get("sell_mean", 0.0) or 0.0)

        def fq_score(mean_ticks: float) -> float:
            denom = max(1.0, float(gate_ticks))
            x = float(mean_ticks) / denom
            return max(-1.0, min(1.0, x))

        w_fq = self._clamp_float(float(self._side_fq_weight), 0.0, 1.0) if bool(self._side_use_fillq) else 0.0
        w_pnl = 1.0 - w_fq

        b_sc = w_pnl * pnl_score(b_avg, b_n)
        s_sc = w_pnl * pnl_score(s_avg, s_n)
        if bool(self._side_use_fillq) and fq_n >= int(self._fq_min_n):
            b_sc += w_fq * fq_score(fq_b)
            s_sc += w_fq * fq_score(fq_s)

        def to_mult(score: float) -> float:
            m = 1.0 + float(score)
            return self._clamp_float(m, float(self._side_perf_min_mult), float(self._side_perf_max_mult))

        buy_mult = to_mult(b_sc)
        sell_mult = to_mult(s_sc)

        def gate_mult(m: float) -> float:
            if m >= 1.0:
                return 1.0
            mx = max(float(self._side_gate_max_mult), 1.0)
            return 1.0 + (1.0 - float(m)) * (mx - 1.0)

        buy_gate_mult = gate_mult(buy_mult)
        sell_gate_mult = gate_mult(sell_mult)

        def inside_adj(m: float) -> int:
            mx = max(0, int(self._side_inside_adj_max))
            if mx <= 0:
                return 0
            if m >= 1.0:
                # mild extra aggressiveness for good side
                return int(round(min(float(mx) * 0.5, (float(m) - 1.0) * float(mx) * 0.5)))
            # widen bad side
            return -int(round((1.0 - float(m)) * float(mx)))

        buy_inside_adj = inside_adj(buy_mult)
        sell_inside_adj = inside_adj(sell_mult)

        dbg = {
            "b_net": b_net, "s_net": s_net,
            "b_n": b_n, "s_n": s_n,
            "b_avg": b_avg, "s_avg": s_avg,
            "fq_n": fq_n, "fq_b": fq_b, "fq_s": fq_s,
            "buy_gate_mult": buy_gate_mult, "sell_gate_mult": sell_gate_mult,
            "buy_inside_adj": buy_inside_adj, "sell_inside_adj": sell_inside_adj,
        }
        return buy_mult, sell_mult, dbg

    def _dd_mult(self, equity: float) -> tuple[float, float]:
        """Return (dd, dd_mult) based on equity peak."""
        if self.state.equity_peak <= 0:
            self.state.equity_peak = float(equity)
        self.state.equity_peak = max(float(self.state.equity_peak), float(equity))
        peak = max(float(self.state.equity_peak), 1e-9)
        dd = max(0.0, (peak - float(equity)) / peak)

        cutoff = max(float(self._size_dd_cutoff), 1e-9)
        t = min(1.0, dd / cutoff)
        dd_mult = 1.0 - t * (1.0 - float(self._size_dd_min_mult))
        dd_mult = self._clamp_float(dd_mult, float(self._size_dd_min_mult), 1.0)
        return dd, dd_mult

    def _edge_ticks_est(self, spread_ticks: int, fq: dict, ml_dir: int, ml_edge: float) -> tuple[float, float]:
        """Estimate per-side edge in ticks using fillQ when warm, otherwise conservative ML fallback."""
        n = int(fq.get("n", 0) or 0)
        if n >= int(self._fq_min_n):
            b = max(0.0, float(fq.get("buy_mean", 0.0) or 0.0))
            s = max(0.0, float(fq.get("sell_mean", 0.0) or 0.0))
            return b, s

        # ML fallback: map edge (abs(p-0.5)) into a few ticks scale.
        base = max(1.0, float(spread_ticks))
        # Normalize ML edge by one-sided threshold to avoid oversizing.
        denom = max(float(self._ml_one_side_edge), float(self._ml_min_edge), 1e-9)
        est = max(0.0, float(ml_edge)) / denom
        est = min(1.0, est)
        ticks = est * base
        if int(ml_dir) > 0:
            return ticks, 0.0
        if int(ml_dir) < 0:
            return 0.0, ticks
        return 0.5 * ticks, 0.5 * ticks

    async def _step(self) -> None:
        snap = await self.store.snapshot()

        c = snap["connectivity"]
        if c["kill_switch"] or not c["bot_running"] or not c["ws_connected"]:
            return

        now_ms = int(snap["ts_ms"])

        st = snap.get("stats", {}) or {}


        # Level 1 throttles: may extend cooldown, reduce size, and rate-limit entries
        qty_mult, quote_interval_ms, entry_mult = self._apply_level1_throttles(snap, now_ms)

        if now_ms < int(self.state.cooldown_until_ms):
            return

        bb = snap["market"]["best_bid"]
        ba = snap["market"]["best_ask"]
        if bb is None or ba is None:
            return

        tick = float(snap["market"]["tick_size"] or 0.1)
        spread = float(snap["market"]["spread"] or 0.0)
        vol = float(snap["market"]["volatility"] or 0.0)
        imb = float(snap["market"]["imbalance"] or 0.0)
        pressure = float(snap["market"]["trade_pressure"] or 0.0)

        # avoid extremely volatile conditions (simple proxy)
        if vol > float(self._max_vol):
            return

        spread_ticks = int(round(spread / tick)) if tick > 0 else 0
        if spread_ticks < int(self.s.STRATEGY_MIN_SPREAD_TICKS):
            return

        pos = float(snap["account"]["position_qty"])
        # Track position open time for time-stop (more accurate than "order placement" time).
        if abs(float(self.state.last_pos_qty)) <= 1e-12 and abs(float(pos)) > 1e-12:
            self.state.pos_open_ts_ms = int(now_ms)
        if abs(float(pos)) <= 1e-12:
            self.state.pos_open_ts_ms = 0
        self.state.last_pos_qty = float(pos)

        in_pos = abs(pos) > 1e-12

        # Inventory ratio used for skew / ML features.
        inv_max = max(float(self._mm_inv_max_pos_qty), 1e-9)
        inv_ratio = self._clamp_float(float(pos) / inv_max, -1.0, 1.0)

        # Compute regime & fill-quality mods every step (needed for smart requote).
        mid = float(snap["market"].get("mid") or ((float(bb) + float(ba)) / 2.0))

        # --- Phase 1: fee-aware economics ---
        # Estimate round-trip maker fees in ticks (open + close).
        fee_bps = float(getattr(self.s, "MAKER_FEE_BPS", 0.0) or 0.0)
        fee_rate = max(0.0, fee_bps / 10000.0)
        fee_ticks = 0
        if float(mid) > 0 and float(tick) > 0 and float(fee_rate) > 0:
            # break-even price move to cover fees: 2*fee_rate*mid
            fee_ticks = int(math.ceil((2.0 * float(fee_rate) * float(mid)) / float(tick)))
        fee_ticks = max(0, int(fee_ticks))
        min_inside_ticks = max(0, int(fee_ticks) + int(self._fee_buffer_ticks))
        reg = self._update_regime(now_ms, float(bb), float(ba), float(tick), float(mid), float(imb))

        spread_to_vol = 0.0
        if mid > 0 and vol > 0:
            spread_to_vol = float(spread) / max(float(mid) * float(vol), 1e-12)

        toxic = (
            float(reg.get("churn_ps", 0.0)) > float(self._regime_max_churn_ps)
            or float(reg.get("imb_std", 0.0)) > float(self._regime_max_imb_std)
            or (vol > 0 and spread_to_vol > 0 and spread_to_vol < float(self._regime_min_spread_to_vol))
        )

        trend_ticks = float(reg.get("trend_ticks", 0.0))
        allow_buy = True
        allow_sell = True
        regime_class = "NORMAL"
        if toxic:
            regime_class = "TOXIC"
        elif abs(trend_ticks) >= float(self._regime_trend_ticks):
            if trend_ticks > 0:
                regime_class = "TREND_UP"
                allow_sell = False
            elif trend_ticks < 0:
                regime_class = "TREND_DOWN"
                allow_buy = False

        # --- Level 4 bandit: select conservative quoting arm (if enabled) ---
        bandit_arm = None
        if self._bandit is not None:
            try:
                net_now = float(st.get("net_pnl_sum", 0.0) or 0.0)
                if int(self._bandit_last_switch_ms) == 0:
                    # initialize baseline and select initial arm
                    self._bandit_last_net = net_now
                    bandit_arm, _ = self._bandit.maybe_switch(int(now_ms), 0.0)
                    self._bandit_last_switch_ms = int(now_ms)
                else:
                    reward = float(net_now) - float(self._bandit_last_net)
                    bandit_arm, switched = self._bandit.maybe_switch(int(now_ms), float(reward))
                    if switched:
                        self._bandit_last_net = net_now
                        self._bandit_last_switch_ms = int(now_ms)
            except Exception:
                # never let bandit break trading loop
                bandit_arm = None

        # Effective knobs (arm overrides, still maker-only / safe-by-default).
        eff_base_inside = int(bandit_arm.base_inside_ticks) if bandit_arm else int(self._mm_base_inside)
        eff_sig_skew_ticks = int(bandit_arm.signal_skew_ticks) if bandit_arm else int(self._mm_signal_skew_ticks)
        eff_requote_price_ticks = int(bandit_arm.requote_price_ticks) if bandit_arm else int(self._mm_requote_price_ticks)
        eff_requote_min_ms = int(bandit_arm.requote_min_ms) if bandit_arm else int(self._mm_requote_min_ms)
        eff_qty_mult_bandit = float(bandit_arm.qty_mult) if bandit_arm else 1.0
        eff_entry_score = float(self._entry_score) * float(entry_mult) * (float(bandit_arm.min_edge_mult) if bandit_arm else 1.0)

        # --- Level 4 micro-ML: online direction predictor (train + signal) ---
        ml_h = 0
        ml_p_up = 0.5
        ml_edge = 0.0
        ml_n = 0
        ml_dir = 0
        ml_skew = 0

        if self._ml is not None:
            try:
                # train matured probes using current mid
                self._ml.update_with_current_mid(int(now_ms), float(mid))

                micro = float(snap["market"].get("microprice") or float(mid))
                x_raw = [
                    float(imb),
                    float(pressure),
                    float(spread_ticks),
                    float(vol),
                    float((micro - float(mid)) / float(tick)) if float(tick) > 0 else 0.0,
                    float(trend_ticks),
                    float(reg.get("churn_ps", 0.0) or 0.0),
                    float(inv_ratio),
                ]

                if (int(now_ms) - int(self.state.last_ml_probe_ts_ms)) >= int(self._ml_probe_interval_ms):
                    self._ml.observe(int(now_ms), float(mid), x_raw)
                    self.state.last_ml_probe_ts_ms = int(now_ms)

                ml_h, ml_p_up, ml_edge, ml_n = self._ml.best_signal(x_raw, int(self._ml_prefer_h), int(self._ml_warmup_n))

                if int(ml_n) >= int(self._ml_warmup_n) and float(ml_edge) >= float(self._ml_min_edge):
                    ml_dir = 1 if float(ml_p_up) > 0.5 else -1
                    strength = min(1.0, float(ml_edge) / max(float(self._ml_one_side_edge), float(self._ml_min_edge), 1e-9))
                    ml_skew = int(round(float(self._ml_skew_ticks) * float(strength)))
            except Exception:
                ml_h, ml_p_up, ml_edge, ml_n, ml_dir, ml_skew = 0, 0.5, 0.0, 0, 0, 0

        # Strategy meta is pushed later (after fee-aware adjustments)

        # Fill-quality feedback: penalize sides with negative mean outcome.
        fq = self._fillq_side_mods(snap)
        buy_thr_mult = float(fq.get("buy_thr_mult", 1.0))
        sell_thr_mult = float(fq.get("sell_thr_mult", 1.0))
        buy_int_ms = int(fq.get("buy_int_ms", 0) or 0)
        sell_int_ms = int(fq.get("sell_int_ms", 0) or 0)
        # --- Phase 1/2: edge estimates in ticks (fillQ warm -> use it; otherwise ML fallback) ---
        edge_buy_ticks, edge_sell_ticks = self._edge_ticks_est(int(spread_ticks), fq, int(ml_dir), float(ml_edge))

        # Normalize signal into [-1..1] for skew (bandit may tune entry_score/skew).
        score = 0.9 * imb + 0.1 * pressure
        sig = 0.0
        if float(eff_entry_score) > 0:
            sig = float(score) / max(float(eff_entry_score), 1e-9)
        sig = self._clamp_float(sig, -1.0, 1.0)
        sig_ticks = int(round(sig * float(eff_sig_skew_ticks)))

        # Convert fillQ multipliers into "widen" ticks: larger multiplier => less aggressive quoting.
        widen_buy = max(0, int(round((max(1.0, buy_thr_mult) - 1.0) * float(self._mm_fq_widen_ticks))))
        widen_sell = max(0, int(round((max(1.0, sell_thr_mult) - 1.0) * float(self._mm_fq_widen_ticks))))

        # Inventory skew (inv_ratio computed once above).
        inv_ticks = int(round(abs(inv_ratio) * float(self._mm_inv_skew_ticks)))

        # Base inside ticks (maker-only). Bandit may override per arm.
        base_inside = int(eff_base_inside)
        max_inside = max(0, int(self._mm_max_inside))

        # Desired inside ticks per side with signal + inventory skew.
        bid_inside = base_inside + sig_ticks
        ask_inside = base_inside - sig_ticks

        # Inventory control: when long -> ease sells, tighten buys; when short -> opposite.
        if float(pos) > 0:
            bid_inside -= inv_ticks
            ask_inside += inv_ticks
        elif float(pos) < 0:
            bid_inside += inv_ticks
            ask_inside -= inv_ticks

        # Fill-quality: widen (less aggressive) the side with negative mean.
        bid_inside -= int(widen_buy)
        ask_inside -= int(widen_sell)

        # ML skew: if model has signal, skew inside ticks toward predicted direction.
        if int(ml_dir) != 0 and int(ml_skew) > 0:
            bid_inside += int(ml_dir) * int(ml_skew)
            ask_inside -= int(ml_dir) * int(ml_skew)

        bid_inside = self._clamp_int(bid_inside, 0, max_inside)
        ask_inside = self._clamp_int(ask_inside, 0, max_inside)

        # If toxic: cancel/rest unless we need to reduce inventory.
        open_orders = [o for o in snap.get("orders", []) if o.get("status") in ("NEW", "PARTIALLY_FILLED")]

        # Helper: in "exit/unwind" situations, NEVER spam cancel_all()+replace every loop.
        # We keep (or gently requote) a single reducing-side order and cancel the opposite side only.
        async def _ensure_exit(exit_side: str, px: float, qty: float) -> None:
            if qty <= 0:
                return

            # Partition open orders by side
            by_side = {"BUY": [], "SELL": []}
            for oo in open_orders:
                s = str(oo.get("side", ""))
                if s in by_side:
                    by_side[s].append(oo)

            def pick_oldest(os: list[dict]) -> Optional[dict]:
                if not os:
                    return None
                return min(os, key=lambda x: int(x.get("ts_ms", 0) or 0))

            existing = pick_oldest(by_side.get(exit_side, []))
            other_side = "SELL" if exit_side == "BUY" else "BUY"

            # Cancel opposite-side orders (we don't want to fight our unwind)
            for oo in by_side.get(other_side, []):
                oid = str(oo.get("order_id") or oo.get("id") or "")
                if oid:
                    await self.engine.cancel_order(oid)

            # Place if none.
            if not existing:
                await self.engine.place_limit_post_only(exit_side, price=float(px), qty=float(qty))
                return

            # Gentle smart-requote (per-side pacing)
            oid = str(existing.get("order_id") or existing.get("id") or "")
            o_px = float(existing.get("price", 0.0) or 0.0)
            o_ts = int(existing.get("ts_ms", 0) or 0)
            age_ms = int(now_ms) - int(o_ts)
            diff_ticks = abs(float(px) - float(o_px)) / float(tick) if float(tick) > 0 else 0.0

            last_rq = int(self.state.last_requote_buy_ts_ms if exit_side == "BUY" else self.state.last_requote_sell_ts_ms)
            can_rq = (int(now_ms) - last_rq) >= int(eff_requote_min_ms)

            need = False
            if age_ms > int(self._order_stale_ms):
                need = True
            elif age_ms >= int(self._mm_min_order_life_ms) and diff_ticks >= float(eff_requote_price_ticks):
                need = True


            if need and bool(self._mm_sticky_quotes) and float(eff_requote_price_ticks) > 0:
                # Sticky quotes: avoid cancel/replace on small drifts. Allow chasing only when far.
                # NOTE: use exit_side (local var), not outer "side".
                chase = False
                if exit_side == "BUY":
                    chase = float(px) > float(o_px)
                else:
                    chase = float(px) < float(o_px)
                mult = 2.0 if chase else 3.0
                if float(diff_ticks) < float(eff_requote_price_ticks) * float(mult):
                    need = False
            if need and can_rq and oid:
                await self.engine.cancel_order(oid)
                if exit_side == "BUY":
                    self.state.last_requote_buy_ts_ms = int(now_ms)
                else:
                    self.state.last_requote_sell_ts_ms = int(now_ms)
                await self.engine.place_limit_post_only(exit_side, price=float(px), qty=float(qty))

        # Hard risk exits (maker-only) still apply.
        if in_pos:
            avg = float(snap["account"].get("avg_entry_price") or 0.0)
            mark = float(snap["market"].get("mark_price") or 0.0)
            if avg > 0 and mark > 0:
                delta = (mark - avg) if pos > 0 else (avg - mark)
                delta_ticks = int(round(delta / tick)) if tick > 0 else 0
                held_ms = now_ms - int(self.state.pos_open_ts_ms or now_ms)

                # Stop-loss/time-stop/take-profit: only quote the reducing side, as aggressively as allowed.
                # Fee-aware TP: do not take tiny profits that can't cover round-trip fees.
                tp_ticks_eff = max(int(self.s.STRATEGY_TAKE_PROFIT_TICKS), int(min_inside_ticks))
                if (
                    delta_ticks >= int(tp_ticks_eff)
                    or delta_ticks <= -int(self.s.STRATEGY_STOP_LOSS_TICKS)
                    or held_ms >= int(self.s.STRATEGY_TIME_STOP_MS)
                ):
                    exit_side = "SELL" if pos > 0 else "BUY"
                    exit_inside = max_inside
                    exit_price = self._maker_price_inside(exit_side, float(bb), float(ba), float(tick), int(exit_inside))
                    # Keep a single reducing order, cancel opposite side only, and requote gently.
                    await _ensure_exit(exit_side, float(exit_price), abs(float(pos)))
                    if delta_ticks <= -int(self.s.STRATEGY_STOP_LOSS_TICKS):
                        self.state.cooldown_until_ms = now_ms + int(self._cooldown_after_stop_ms)
                    # remember regime for observability
                    self.state.last_regime_class = str(regime_class)
                    return

        # If regime is toxic: do not open new inventory. If holding, only reduce; if flat, cancel and wait.
        if regime_class == "TOXIC":
            if not in_pos:
                # In toxic conditions while flat: pull quotes and wait.
                if open_orders:
                    await self.engine.cancel_all()

                # update dashboard meta even while resting
                if (int(now_ms) - int(self.state.last_meta_ts_ms)) >= 250:
                    equity = float(snap["account"].get("equity_usdt", 0.0) or 0.0)
                    dd, _dd_mult = self._dd_mult(float(equity))
                    buy_side_mult, sell_side_mult, _sp = self._side_perf_mults(int(now_ms))
                    max_notional = max(0.0, float(equity) * max(0.0, float(self._max_notional_pct)))
                    max_entry_qty = (max_notional / float(mid)) if float(mid) > 0 else 0.0
                    inside_used = int(bid_inside) + int(ask_inside)
                    await self.store.set_strategy_meta({
                        "regime_state": str(regime_class),
                        "fee_ticks": int(fee_ticks),
                        "min_inside_ticks": int(min_inside_ticks),
                        "inside_used": int(inside_used),
                        "size_mult": float(self.state.size_mult),
                        "side_mult_buy": float(buy_side_mult),
                        "side_mult_sell": float(sell_side_mult),
                        "dd_now": float(dd),
                        "max_notional": float(max_notional),
                        "max_entry_qty": float(max_entry_qty),
                        "max_notional_pct": float(self._max_notional_pct),
                        "ml_h": int(ml_h),
                        "ml_p_up": float(ml_p_up),
                        "ml_edge": float(ml_edge),
                        "ml_n": int(ml_n),
                        "bandit_arm": str(bandit_arm.name) if bandit_arm else "",
                        "bandit_eps": float(self._bandit.epsilon) if self._bandit is not None else 0.0,
                    })
                    self.state.last_meta_ts_ms = int(now_ms)

                self.state.last_regime_class = str(regime_class)
                return

            # In toxic conditions while holding: only unwind (do not spam cancel_all).
            exit_side = "SELL" if pos > 0 else "BUY"
            exit_price = self._maker_price_inside(exit_side, float(bb), float(ba), float(tick), int(max_inside))
            await _ensure_exit(exit_side, float(exit_price), abs(float(pos)))

            # update dashboard meta even during unwind-only mode
            if (int(now_ms) - int(self.state.last_meta_ts_ms)) >= 250:
                equity = float(snap["account"].get("equity_usdt", 0.0) or 0.0)
                dd, _dd_mult = self._dd_mult(float(equity))
                buy_side_mult, sell_side_mult, _sp = self._side_perf_mults(int(now_ms))
                max_notional = max(0.0, float(equity) * max(0.0, float(self._max_notional_pct)))
                max_entry_qty = (max_notional / float(mid)) if float(mid) > 0 else 0.0
                inside_used = int(bid_inside) + int(ask_inside)
                await self.store.set_strategy_meta({
                    "regime_state": str(regime_class),
                    "fee_ticks": int(fee_ticks),
                    "min_inside_ticks": int(min_inside_ticks),
                    "inside_used": int(inside_used),
                    "size_mult": float(self.state.size_mult),
                    "side_mult_buy": float(buy_side_mult),
                    "side_mult_sell": float(sell_side_mult),
                    "dd_now": float(dd),
                    "max_notional": float(max_notional),
                    "max_entry_qty": float(max_entry_qty),
                    "max_notional_pct": float(self._max_notional_pct),
                    "ml_h": int(ml_h),
                    "ml_p_up": float(ml_p_up),
                    "ml_edge": float(ml_edge),
                    "ml_n": int(ml_n),
                    "bandit_arm": str(bandit_arm.name) if bandit_arm else "",
                    "bandit_eps": float(self._bandit.epsilon) if self._bandit is not None else 0.0,
                })
                self.state.last_meta_ts_ms = int(now_ms)

            self.state.last_regime_class = str(regime_class)
            return

        # Fallback: if Level-3 disabled, keep old behavior (rare).
        if not self._mm_enabled:
            return

        # Determine which sides we *should* quote now.
        want_buy = allow_buy
        want_sell = allow_sell

        # Inventory gate: do not add beyond target max; always allow reducing side when in position.
        if in_pos:
            if pos > 0:
                want_sell = True  # reducing long
                # block adding long if near max
                if pos >= inv_max - 1e-12:
                    want_buy = False
            elif pos < 0:
                want_buy = True  # reducing short
                if abs(pos) >= inv_max - 1e-12:
                    want_sell = False

        # Extreme fillQ penalty: optionally stop quoting that side (still allow reducing).
        denom = max(float(self._fq_neg_entry_mult_max) - 1.0, 1e-9)
        p_buy = (max(1.0, buy_thr_mult) - 1.0) / denom
        p_sell = (max(1.0, sell_thr_mult) - 1.0) / denom
        if not in_pos and p_buy > 0.90:
            want_buy = False
        if not in_pos and p_sell > 0.90:
            want_sell = False

        # ML gating: when signal is strong, optionally go one-sided while flat;
        # in-position, block *adding* against the model direction (but never block reducing).
        if int(ml_dir) != 0 and int(ml_n) >= int(self._ml_warmup_n) and float(ml_edge) >= float(self._ml_min_edge):
            if not in_pos and float(ml_edge) >= float(self._ml_one_side_edge):
                if int(ml_dir) > 0:
                    want_sell = False
                else:
                    want_buy = False
            if in_pos:
                if float(pos) > 0 and int(ml_dir) < 0:
                    want_buy = False
                if float(pos) < 0 and int(ml_dir) > 0:
                    want_sell = False

        # --- Phase 1: fee-aware gate ---
        # When flat, do not open inventory if estimated edge < fees+buffer.
        # When in position, always allow reduce-side; add-side only with stronger edge.
        reduce_side = None
        add_side = None
        if in_pos:
            reduce_side = "SELL" if float(pos) > 0 else "BUY"
            add_side = "BUY" if float(pos) > 0 else "SELL"

        # Strict economics uses min_inside_ticks (= fees+buffer). On high-priced symbols,
        # this can exceed the prevailing spread, leading to near-zero trades on testnet.
        # If enabled, relax the *gate threshold* to a fraction of current spread while
        # keeping fee_ticks/min_inside_ticks observable.
        gate_ticks = int(min_inside_ticks)
        if bool(self._relax_fee_gate):
            relaxed = max(1, int(float(spread_ticks) * float(self._relax_gate_spread_mult)))
            gate_ticks = min(int(gate_ticks), int(relaxed))

        # Phase 2c: side performance control (BUY/SELL separately)
        buy_side_mult, sell_side_mult, sp = self._side_perf_mults(int(now_ms), fq=fq, gate_ticks=int(gate_ticks))
        buy_gate_mult = float(sp.get("buy_gate_mult", 1.0))
        sell_gate_mult = float(sp.get("sell_gate_mult", 1.0))
        buy_inside_adj = int(sp.get("buy_inside_adj", 0))
        sell_inside_adj = int(sp.get("sell_inside_adj", 0))

        # If raw spread is too small, do not open (but allow reduce-side).
        if float(spread_ticks) < float(gate_ticks):
            if not in_pos:
                want_buy = False
                want_sell = False
            else:
                if add_side == "BUY":
                    want_buy = False
                elif add_side == "SELL":
                    want_sell = False

        if not in_pos:
            if float(edge_buy_ticks) < float(gate_ticks) * float(buy_gate_mult):
                want_buy = False
            if float(edge_sell_ticks) < float(gate_ticks) * float(sell_gate_mult):
                want_sell = False
        else:
            add_req = float(int(gate_ticks) + int(self._fee_add_extra_ticks))
            if add_side == "BUY" and want_buy and float(edge_buy_ticks) < add_req * float(buy_gate_mult):
                want_buy = False
            if add_side == "SELL" and want_sell and float(edge_sell_ticks) < add_req * float(sell_gate_mult):
                want_sell = False


        # Phase 2c: widen/tighten per-side inside a bit based on recent per-side quality.
        # Negative adj widens (less inside) for the worse side.
        if int(buy_inside_adj) != 0:
            bid_inside = self._clamp_int(int(bid_inside) + int(buy_inside_adj), 0, int(max_inside))
        if int(sell_inside_adj) != 0:
            ask_inside = self._clamp_int(int(ask_inside) + int(sell_inside_adj), 0, int(max_inside))


        # --- Phase 1: fee-aware min inside ---
        # Ensure two-sided quoting is not tighter than fees+buffer.
        if want_buy and want_sell and int(min_inside_ticks) > 0:
            max_sum_inside = int(spread_ticks) - int(min_inside_ticks)
            if max_sum_inside < 0:
                max_sum_inside = 0
            sum_inside = int(bid_inside) + int(ask_inside)
            if sum_inside > max_sum_inside:
                excess = sum_inside - max_sum_inside
                if in_pos:
                    # reduce add-side first; keep reduce-side as aggressive as possible
                    if float(pos) > 0:
                        take = min(int(bid_inside), int(excess))
                        bid_inside -= take
                        excess -= take
                    elif float(pos) < 0:
                        take = min(int(ask_inside), int(excess))
                        ask_inside -= take
                        excess -= take
                for _ in range(int(excess)):
                    if bid_inside >= ask_inside and bid_inside > 0:
                        bid_inside -= 1
                    elif ask_inside > 0:
                        ask_inside -= 1
                    elif bid_inside > 0:
                        bid_inside -= 1

            bid_inside = self._clamp_int(bid_inside, 0, max_inside)
            ask_inside = self._clamp_int(ask_inside, 0, max_inside)

        inside_used = int(bid_inside) + int(ask_inside)

        # --- Phase 2 precompute: dynamic sizing + side performance (for qty + dashboard) ---
        equity = float(snap["account"].get("equity_usdt", 0.0) or 0.0)
        dd, dd_mult = self._dd_mult(float(equity))
        # side multipliers were computed earlier (Phase 2c), after gate_ticks was known

        edge_best = max(float(edge_buy_ticks), float(edge_sell_ticks))
        denom = max(1.0, float(gate_ticks))
        r = (edge_best / denom) / max(float(self._size_edge_full_at), 1e-9)
        r = max(0.0, min(1.0, r))
        edge_mult = float(self._size_min_mult) + (float(self._size_max_mult) - float(self._size_min_mult)) * (r ** float(self._size_edge_exp))

        regime_mult = 1.0
        if str(regime_class) == "TOXIC":
            regime_mult = float(self._size_regime_toxic_mult)
        elif str(regime_class).startswith("TREND_"):
            regime_mult = float(self._size_regime_trend_mult)

        target_size = float(edge_mult) * float(dd_mult) * float(regime_mult)
        target_size = self._clamp_float(target_size, float(self._size_min_mult), float(self._size_max_mult))

        cur = float(self.state.size_mult)
        if target_size <= cur:
            cur = target_size
        else:
            if (int(now_ms) - int(self.state.size_last_inc_ms)) >= int(self._size_grow_cooldown_ms):
                cur = min(target_size, cur + float(self._size_grow_step))
                self.state.size_last_inc_ms = int(now_ms)
        self.state.size_mult = float(self._clamp_float(cur, float(self._size_min_mult), float(self._size_max_mult)))

        max_notional = max(0.0, float(equity) * max(0.0, float(self._max_notional_pct)))
        max_entry_qty_by_notional = (max_notional / float(mid)) if float(mid) > 0 else 0.0

        # Phase-2 sizing seed: either ORDER_QTY or an equity-based base-notional percentage.
        base_seed_qty = float(self.s.ORDER_QTY)
        if float(self._base_notional_pct) > 0 and float(mid) > 0 and float(equity) > 0:
            base_seed_qty = max(0.0, float(equity) * float(self._base_notional_pct) / float(mid))

        # Additional safety: cap by a multiple of the seed qty.
        max_entry_qty_by_mult = float(base_seed_qty) * float(self._size_max_qty_mult)
        max_entry_qty = min(max_entry_qty_by_notional if max_entry_qty_by_notional > 0 else max_entry_qty_by_mult, max_entry_qty_by_mult)
        max_entry_qty = max(0.0, float(max_entry_qty))

        buy_int_factor = self._clamp_float(1.0 / max(float(buy_side_mult), 0.25), 1.0, 3.0)
        sell_int_factor = self._clamp_float(1.0 / max(float(sell_side_mult), 0.25), 1.0, 3.0)

        # Push a small meta blob to dashboard (throttled).
        if (int(now_ms) - int(self.state.last_meta_ts_ms)) >= 250:
            await self.store.set_strategy_meta({
                "regime_state": str(regime_class),
                "fee_ticks": int(fee_ticks),
                "min_inside_ticks": int(min_inside_ticks),
                "gate_ticks": int(gate_ticks),
                "inside_used": int(inside_used),
                "size_mult": float(self.state.size_mult),
                "base_seed_qty": float(base_seed_qty),
                "base_notional_pct": float(self._base_notional_pct),
                "side_mult_buy": float(buy_side_mult),
                "side_mult_sell": float(sell_side_mult),
                "side_gate_buy": float(buy_gate_mult),
                "side_gate_sell": float(sell_gate_mult),
                "side_inside_adj_buy": int(buy_inside_adj),
                "side_inside_adj_sell": int(sell_inside_adj),
                "side_pnl_avg_buy": float(sp.get("b_avg", 0.0) or 0.0),
                "side_pnl_avg_sell": float(sp.get("s_avg", 0.0) or 0.0),
                "side_trades_buy": int(sp.get("b_n", 0) or 0),
                "side_trades_sell": int(sp.get("s_n", 0) or 0),
                "side_fq_n": int(sp.get("fq_n", 0) or 0),
                "side_fq_mean_buy": float(sp.get("fq_b", 0.0) or 0.0),
                "side_fq_mean_sell": float(sp.get("fq_s", 0.0) or 0.0),
                "dd_now": float(dd),
                "max_notional": float(max_notional),
                "max_entry_qty": float(max_entry_qty),
                "max_notional_pct": float(self._max_notional_pct),
                "ml_h": int(ml_h),
                "ml_p_up": float(ml_p_up),
                "ml_edge": float(ml_edge),
                "ml_n": int(ml_n),
                "bandit_arm": str(bandit_arm.name) if bandit_arm else "",
                "bandit_eps": float(self._bandit.epsilon) if self._bandit is not None else 0.0,
            })
            self.state.last_meta_ts_ms = int(now_ms)

        if not want_buy and not want_sell:
            # Cancel resting orders if we shouldn't quote either side.
            if open_orders:
                # Gate-hold: keep orders briefly even if filters flip off, to avoid cancel-storms.
                try:
                    oldest = min(int(oo.get("ts_ms", 0) or 0) for oo in open_orders)
                except Exception:
                    oldest = 0
                if oldest > 0 and (int(now_ms) - oldest) < int(self._gate_hold_ms):
                    return
                await self.engine.cancel_all()
            return

        # Compute desired prices.
        bid_px = self._maker_price_inside("BUY", float(bb), float(ba), float(tick), int(bid_inside))
        ask_px = self._maker_price_inside("SELL", float(bb), float(ba), float(tick), int(ask_inside))
        # Base qty with Level-1 throttle + bandit + dynamic sizing
        base_qty = max(0.0, float(base_seed_qty) * float(qty_mult) * float(eff_qty_mult_bandit) * float(self.state.size_mult))
        if base_qty <= 0:
            return

        bid_qty = min(float(base_qty) * float(buy_side_mult), float(max_entry_qty)) if float(max_entry_qty) > 0 else float(base_qty) * float(buy_side_mult)
        ask_qty = min(float(base_qty) * float(sell_side_mult), float(max_entry_qty)) if float(max_entry_qty) > 0 else float(base_qty) * float(sell_side_mult)

        # Inventory control: reduce-side can close full position; add-side stays capped.
        # Inventory control: reduce-side can close full position; add-side stays capped.
        if in_pos:
            if pos > 0:
                # long: SELL reduces, BUY adds
                ask_qty = max(float(ask_qty), abs(float(pos)))
                bid_qty = float(bid_qty) * max(0.0, float(self._mm_inv_add_qty_mult))
            elif pos < 0:
                # short: BUY reduces, SELL adds
                bid_qty = max(float(bid_qty), abs(float(pos)))
                ask_qty = float(ask_qty) * max(0.0, float(self._mm_inv_add_qty_mult))

        # Smart requote: update/cancel per-side only when needed (reduces churn).
        by_side = {"BUY": [], "SELL": []}
        for o in open_orders:
            s = str(o.get("side", ""))
            if s in by_side:
                by_side[s].append(o)

        def pick_one(os: list[dict]) -> Optional[dict]:
            if not os:
                return None
            # prefer oldest (more likely to be stale)
            return min(os, key=lambda x: int(x.get("ts_ms", 0) or 0))

        buy_o = pick_one(by_side["BUY"])
        sell_o = pick_one(by_side["SELL"])

        async def ensure(side: str, want: bool, px: float, qty: float) -> None:
            nonlocal buy_o, sell_o
            existing = buy_o if side == "BUY" else sell_o
            if not want or qty <= 0:
                if existing:
                    oid = str(existing.get("order_id") or existing.get("id") or "")
                    if oid:
                        await self.engine.cancel_order(oid)
                        if side == "BUY":
                            self.state.last_requote_buy_ts_ms = int(now_ms)
                        else:
                            self.state.last_requote_sell_ts_ms = int(now_ms)
                return

            # If no existing, place a new one (respect soft-stop global pacing on initial placement).
            if not existing:
                side_last = int(self.state.last_quote_buy_ts_ms) if side == "BUY" else int(self.state.last_quote_sell_ts_ms)
                side_min_int = int(max(int(quote_interval_ms), int(buy_int_ms) if side == "BUY" else int(sell_int_ms)))
                if side_min_int > 0:
                    side_min_int = int(round(float(side_min_int) * (float(buy_int_factor) if side == "BUY" else float(sell_int_factor))))
                if side_min_int > 0 and (int(now_ms) - side_last) < side_min_int:
                    return
                o = await self.engine.place_limit_post_only(side, price=float(px), qty=float(qty))
                if o.status == "NEW":
                    self.state.last_quote_ts_ms = int(now_ms)
                    if side == "BUY":
                        self.state.last_quote_buy_ts_ms = int(now_ms)
                    else:
                        self.state.last_quote_sell_ts_ms = int(now_ms)
                return

            # Decide whether to requote.
            oid = str(existing.get("order_id") or existing.get("id") or "")
            o_px = float(existing.get("price", 0.0) or 0.0)
            o_ts = int(existing.get("ts_ms", 0) or 0)
            age_ms = int(now_ms) - int(o_ts)
            diff_ticks = 0.0
            if tick > 0:
                diff_ticks = abs(float(px) - float(o_px)) / float(tick)

            last_rq = int(self.state.last_requote_buy_ts_ms if side == "BUY" else self.state.last_requote_sell_ts_ms)
            can_rq = (int(now_ms) - last_rq) >= int(eff_requote_min_ms)

            
            need = False
            if age_ms > int(self._order_stale_ms):
                need = True
            elif age_ms >= int(self._mm_min_order_life_ms) and (diff_ticks >= float(eff_requote_price_ticks)):
                # Sticky quotes: require a larger drift before we cancel/replace, especially when chasing.
                need = True
                if bool(self._mm_sticky_quotes):
                    chase = (float(px) > float(o_px)) if side == "BUY" else (float(px) < float(o_px))
                    mult = 2.0 if chase else 3.0
                    if diff_ticks < float(eff_requote_price_ticks) * float(mult):
                        need = False

            if need and can_rq and oid:
                # Do not cancel unless we can immediately replace (prevents cancel-only churn).
                side_last = int(self.state.last_quote_buy_ts_ms) if side == "BUY" else int(self.state.last_quote_sell_ts_ms)
                side_min_int = int(max(int(quote_interval_ms), int(buy_int_ms) if side == "BUY" else int(sell_int_ms)))
                if side_min_int > 0:
                    side_min_int = int(round(float(side_min_int) * (float(buy_int_factor) if side == "BUY" else float(sell_int_factor))))
                if side_min_int > 0 and (int(now_ms) - side_last) < side_min_int:
                    return

                await self.engine.cancel_order(oid)

                if side == "BUY":
                    self.state.last_requote_buy_ts_ms = int(now_ms)
                else:
                    self.state.last_requote_sell_ts_ms = int(now_ms)
                # place replacement (pacing already checked above)
                o2 = await self.engine.place_limit_post_only(side, price=float(px), qty=float(qty))
                if o2.status == "NEW":
                    self.state.last_quote_ts_ms = int(now_ms)
                    if side == "BUY":
                        self.state.last_quote_buy_ts_ms = int(now_ms)
                    else:
                        self.state.last_quote_sell_ts_ms = int(now_ms)

        await ensure("BUY", bool(want_buy), float(bid_px), float(bid_qty))
        await ensure("SELL", bool(want_sell), float(ask_px), float(ask_qty))

        # Remember last knobs used (for observability / future UI if needed).
        self.state.last_regime_class = str(regime_class)
        self.state.last_mm_bid_inside = int(bid_inside)
        self.state.last_mm_ask_inside = int(ask_inside)
        return

