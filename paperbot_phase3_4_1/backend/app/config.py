from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings.

    Values can be overridden via environment variables or a .env file.

    IMPORTANT: This project is TESTNET-ONLY by design.
    REST_BASE and WS_BASE are validated in __init__.
    """

    model_config = SettingsConfigDict(env_prefix="", case_sensitive=False)

    # ---- TESTNET ONLY ----
    REST_BASE: str = "https://demo-fapi.binance.com"
    WS_BASE: str = "wss://fstreamWS_BASE_MUST_NOT_CHANGE"  # safety guard (validated in __init__)

    SYMBOL: str = "ETHUSDT"

    # Strategy selection
    #   - "swing_1m" (default): 1–5 minute maker-entry swing
    #   - "scalp_mm": legacy micro-scalp/quoting (maker-only)
    STRATEGY_MODE: str = "swing_1m"
    STREAMS_DEPTH: str = "{symbol_lower}@depth@100ms"
    STREAMS_TRADES: str = "{symbol_lower}@aggTrade"
    STREAMS_MARK: str = "{symbol_lower}@markPrice@1s"

    HOST: str = "127.0.0.1"
    PORT: int = 8000

    # paper trading
    PAPER_START_BALANCE_USDT: float = 1000.0
    LEVERAGE: float = 10.0
    MAKER_FEE_BPS: float = 2.0

    # --- Phase 1: fee-aware quoting / gating ---
    # Buffer in ticks added on top of estimated round-trip maker fees.
    STRATEGY_FEE_BUFFER_TICKS: int = 1
    # When in position, require stronger edge before adding to inventory.
    STRATEGY_FEE_ADD_EXTRA_TICKS: int = 1

    # --- Phase 2: dynamic sizing + side asymmetry ---
    # Entry/add notional cap as % of equity (reduce-side exits are never blocked).
    STRATEGY_MAX_NOTIONAL_PCT: float = 0.10

    # --- Swing (1–5 minute) strategy params ---
    SWING_MIN_EDGE_MULT: float = 1.2
    SWING_ENTRY_BASE_OFFSET_TICKS: int = 8
    SWING_ENTRY_OFFSET_MULT_FEES: float = 0.7
    SWING_ENTRY_OFFSET_MULT_VOL: float = 0.8
    SWING_ENTRY_COOLDOWN_MS: int = 15000

    SWING_MIN_ORDER_LIFE_MS: int = 2000
    SWING_REQUOTE_MIN_MS: int = 4000
    SWING_REQUOTE_DRIFT_TICKS: int = 8
    SWING_GATE_HOLD_MS: int = 2000

    # --- Phase 3.4: exit execution discipline (maker-only) ---
    # How long we keep an exit order resting before we consider repricing.
    SWING_EXIT_PATIENCE_MS: int = 2500
    # Minimum time between exit cancel/replace attempts.
    SWING_EXIT_REQUOTE_MIN_MS: int = 2500
    # Minimum drift in ticks from the desired exit price before repricing.
    SWING_EXIT_REQUOTE_DRIFT_TICKS: int = 1

    SWING_TP_MULT_FEES: float = 2.0
    SWING_TP_TICKS_MIN: int = 16
    SWING_SL_MULT_FEES: float = 1.5
    SWING_SL_TICKS_MIN: int = 12
    SWING_MAX_HOLD_MS: int = 300000  # 5 minutes

    # --- Phase 3.3a: fee-cover → activate trailing profit (let winners run) ---
    # When enabled, the swing strategy will NOT take profit at the first TP threshold.
    # Instead it will *activate* a trailing exit once delta_ticks >= activate_ticks, and
    # exit when profit retraces from peak by SWING_TRAIL_DD_PCT (bounded by min retrace).
    SWING_TRAIL_ENABLE: bool = True
    # Activation threshold in ticks is based on estimated *round-trip maker fees* in ticks
    # (fee_ticks) multiplied by this factor. Recommended 1.5..3.0. Default 2.0.
    SWING_TRAIL_ACTIVATE_MULT_FEES: float = 2.0
    # Absolute floor for activation ticks (helps avoid tiny peaks on low fee_ticks / low vol).
    # For ETH 8 ticks is a reasonable starting point.
    SWING_TRAIL_MIN_ACTIVATE_TICKS: int = 8
    # Trailing drawdown from peak profit (e.g., 0.10 = 10% retrace)
    SWING_TRAIL_DD_PCT: float = 0.10
    # Minimum retrace in ticks to trigger exit (avoids too-tight trailing on small peaks)
    SWING_TRAIL_MIN_RETRACE_TICKS: int = 4
    # Minimum peak in ticks required before we allow trailing-trigger exits
    SWING_TRAIL_MIN_PEAK_TICKS: int = 0

    # --- Phase 3.3c: two-mode trailing (protect then let it run) ---
    # After trailing activates, we first protect profits above breakeven+ (be_floor_ticks),
    # then switch to a wider trailing distance to avoid getting shaken out by noise.
    SWING_TRAIL_BE_MULT: float = 1.3
    SWING_TRAIL_BE_BUFFER_TICKS: int = 3
    SWING_TRAIL_LETRUN_AFTER_MS: int = 45_000
    SWING_TRAIL_LETRUN_K_MULT: float = 1.25

    # --- Phase 3.3d: anti-loser exits (early adverse selection + time-decay) ---
    SWING_EARLY_AS_ENABLE: bool = True
    SWING_EARLY_AS_WINDOW_MS: int = 25_000
    SWING_EARLY_AS_LOSS_TICKS: int = 12
    # Trigger early-AS either by toxic_score (regime v2) or churn spikes
    SWING_EARLY_AS_TOXIC_SCORE: float = 1.2
    SWING_EARLY_AS_CHURN_PER_S: float = 10.0
    # Extra inside ticks for early-AS exit (more aggressive but still maker-only bounded by spread)
    SWING_EARLY_AS_INSIDE_ADD_TICKS: int = 1

    SWING_TIME_DECAY_ENABLE: bool = True
    SWING_TIME_DECAY_START_MS: int = 60_000
    SWING_TIME_DECAY_STEP_MS: int = 30_000
    SWING_TIME_DECAY_MAX_LEVEL: int = 3
    SWING_TIME_DECAY_MIN_LOSS_TICKS: int = 6
    SWING_TIME_DECAY_INSIDE_ADD_PER_LEVEL: int = 1
    SWING_TIME_DECAY_MAX_INSIDE_ADD_TICKS: int = 4


    SWING_RISK_PCT: float = 0.005  # 0.5% equity risk per trade
    # Optional: simulated emergency exit inside paper model only (never sent to exchange)
    SWING_EMERGENCY_EXIT: bool = False

    # --- Phase 1.5: minimal execution autotuner (bounded, observable) ---
    AUTOTUNE_ENABLED: bool = True
    AUTOTUNE_INTERVAL_MS: int = 30_000

    AUTOTUNE_OFFSET_ADJ_MIN: int = -6
    AUTOTUNE_OFFSET_ADJ_MAX: int = 12
    AUTOTUNE_COOLDOWN_MIN_MS: int = 5_000
    AUTOTUNE_COOLDOWN_MAX_MS: int = 30_000
    AUTOTUNE_REQUOTE_MIN_MS_MIN: int = 3_000
    AUTOTUNE_REQUOTE_MIN_MS_MAX: int = 12_000

    # Thresholds (simple heuristics)
    AUTOTUNE_REJECT_RATE_HI: float = 0.10
    AUTOTUNE_CANCEL_RATE_HI: float = 0.25
    AUTOTUNE_FILL_RATE_LO: float = 0.15
    AUTOTUNE_EV_OPEN_RATE_HI: float = 0.20

    # Step sizes
    AUTOTUNE_OFFSET_STEP_TICKS: int = 1
    AUTOTUNE_COOLDOWN_STEP_MS: int = 2_000
    AUTOTUNE_REQUOTE_STEP_MS: int = 1_000

    # --- Phase 1.5: safety autopilot (entry-only pauses) ---
    SAFETY_ENABLED: bool = True
    # If TOXIC regime persists longer than this, pause new entries for SAFETY_PAUSE_MS.
    SAFETY_TOXIC_PAUSE_AFTER_MS: int = 300_000
    SAFETY_PAUSE_MS: int = 900_000
    # Optional loss-based safety (disabled by default; set >0 to enable)
    SAFETY_LOSS_LIMIT_1H_PCT: float = 0.0
    SAFETY_EDGE_TIGHTEN_MAX_MULT: float = 1.30

    # --- Phase 2.1: NDJSON recorder (market + decisions) ---
    # Writes local NDJSON files for offline reproduction/backtests.
    # Safe defaults: bounded queue (drops observable), periodic flush, size rotation.
    RECORDER_ENABLED: bool = True
    RECORDER_DIR: str = "recordings"
    RECORDER_FLUSH_MS: int = 2000
    RECORDER_QUEUE_MAX: int = 5000
    RECORDER_MAX_FILE_MB: int = 256
    # Frame sampling interval (strategy loop will downsample recorder frames).
    RECORDER_FRAME_MS: int = 250


    # Base order notional (as % of equity). Used to compute qty = equity*pct/mid
    # before applying size_mult / side_mult. Set to 0 to use ORDER_QTY.
    STRATEGY_BASE_NOTIONAL_PCT: float = 0.05

    # If fees are larger than typical spread on this symbol, strict fee gate can result in near-zero trades.
    # When enabled, we relax the *gate threshold* to a fraction of the current spread (still maker-only),
    # while keeping fee_ticks/min_inside_ticks observable on dashboard.
    STRATEGY_RELAX_FEE_GATE: bool = True
    STRATEGY_RELAX_GATE_SPREAD_MULT: float = 0.50
    # Global size multiplier bounds
    STRATEGY_SIZE_MIN_MULT: float = 0.25
    STRATEGY_SIZE_MAX_MULT: float = 2.50
    # Shape: how quickly size increases with edge
    STRATEGY_SIZE_EDGE_EXP: float = 1.40
    # Normalize edge by (edge/min_inside)/FULL_AT -> 1.0 means max size
    STRATEGY_SIZE_EDGE_FULL_AT: float = 1.00
    # Regime multipliers
    STRATEGY_SIZE_REGIME_TOXIC_MULT: float = 0.50
    STRATEGY_SIZE_REGIME_TREND_MULT: float = 1.10
    # Drawdown based scaling
    STRATEGY_SIZE_DD_CUTOFF: float = 0.01
    STRATEGY_SIZE_DD_MIN_MULT: float = 0.35
    # Cooldown on growth (avoid sudden jumps)
    STRATEGY_SIZE_GROW_COOLDOWN_MS: int = 5000
    STRATEGY_SIZE_GROW_STEP: float = 0.15
    # Absolute cap relative to base qty (prevents huge orders if equity is large)
    STRATEGY_SIZE_MAX_QTY_MULT: float = 3.00

    # Side performance control (BUY/SELL separately)
    STRATEGY_SIDE_PERF_WINDOW_MS: int = 600_000
    STRATEGY_SIDE_PERF_MIN_TRADES: int = 5
    STRATEGY_SIDE_PERF_TARGET_USDT: float = 0.02
    STRATEGY_SIDE_PERF_MIN_MULT: float = 0.30
    STRATEGY_SIDE_PERF_MAX_MULT: float = 1.20

    # Phase 2c: side performance control can also incorporate FillQ mean ticks (per-side)
    # to react faster than realized PnL alone.
    STRATEGY_SIDE_USE_FILLQ: bool = True
    STRATEGY_SIDE_FQ_WEIGHT: float = 0.35  # 0..1, rest is PnL
    STRATEGY_SIDE_GATE_MAX_MULT: float = 1.75  # increase gate threshold for bad side
    STRATEGY_SIDE_INSIDE_ADJ_MAX_TICKS: int = 2  # widen bad side by up to N inside ticks
    STRATEGY_GATE_HOLD_MS: int = 2000  # keep orders briefly even if gate flips off

    ORDER_QTY: float = 0.001

    # --- Strategy defaults (tuned to be more active on testnet) ---
    # NOTE: "profit" cannot be guaranteed. These defaults simply increase
    # trade frequency and improve fill probability in a paper environment.
    STRATEGY_MIN_SPREAD_TICKS: int = 1
    STRATEGY_TAKE_PROFIT_TICKS: int = 1
    STRATEGY_STOP_LOSS_TICKS: int = 2
    STRATEGY_TIME_STOP_MS: int = 15000

    # New knobs to make the bot "trade more" without going taker
    STRATEGY_MAX_VOL: float = 0.003
    STRATEGY_ENTRY_SCORE: float = 0.05
    # IMPORTANT:
    # Our paper fill model infers executions from *exchange* top-of-book size changes.
    # We do NOT insert our own inside-spread orders into the displayed book.
    # Therefore, quoting inside the spread can look "smart" but may never fill in
    # this skeleton. Keep it at 0 unless you also simulate our order in the book.
    STRATEGY_PEG_INSIDE_TICKS: int = 0
    ORDER_STALE_MS: int = 45000
    STRATEGY_COOLDOWN_AFTER_STOP_MS: int = 4000
    # --- Level 1 risk throttles (simple, low-risk) ---
    # Dynamic cooldown after consecutive losing CLOSED trades (net < 0)
    STRATEGY_DYN_COOLDOWN_STREAK: int = 3
    STRATEGY_DYN_COOLDOWN_MIN_MS: int = 10_000
    STRATEGY_DYN_COOLDOWN_MAX_MS: int = 60_000

    # Soft-stop: reduce aggressiveness instead of stopping trading
    STRATEGY_SOFTSTOP_STREAK: int = 2
    STRATEGY_SOFTSTOP_QTY_MULT: float = 0.33
    STRATEGY_SOFTSTOP_QUOTE_INTERVAL_MS: int = 400
    STRATEGY_SOFTSTOP_ENTRY_SCORE_MULT: float = 1.25
    # --- Level 2: regime filter + fill-quality feedback (still simple) ---
    STRATEGY_REGIME_WINDOW_MS: int = 5000
    STRATEGY_REGIME_MAX_CHURN_PER_S: float = 8.0
    STRATEGY_REGIME_MAX_IMB_STD: float = 0.35
    STRATEGY_REGIME_MIN_SPREAD_TO_VOL: float = 0.6
    STRATEGY_REGIME_TREND_TICKS: int = 4

    # --- Phase 3.1: Regime classifier v2 (trend/chop/toxic) ---
    # Uses a rolling mid-price window to compute an efficiency ratio (trend vs noise),
    # plus baseline medians for spread/vol to detect spikes (toxic).
    STRATEGY_REGIME_V2_WINDOW_MS: int = 15_000
    STRATEGY_REGIME_V2_BASELINE_MS: int = 60_000
    # Trend/chop thresholds (efficiency ratio in [0..1])
    STRATEGY_REGIME_V2_TREND_ER_THR: float = 0.38
    STRATEGY_REGIME_V2_CHOP_THR: float = 0.72
    # Toxic spike thresholds (relative to baseline median)
    STRATEGY_REGIME_V2_SPREAD_SPIKE_SOFT: float = 1.6
    STRATEGY_REGIME_V2_VOL_SPIKE_SOFT: float = 1.7
    STRATEGY_REGIME_V2_CHURN_SPIKE_SOFT: float = 1.5
    STRATEGY_REGIME_V2_TOXIC_SOFT_SCORE: float = 1.0
    STRATEGY_REGIME_V2_TOXIC_HARD_SCORE: float = 1.5
    # Soft guards (bounded): tighten gate/offset instead of hard-stopping
    STRATEGY_REGIME_V2_TOXIC_SOFT_EDGE_MULT: float = 1.15
    STRATEGY_REGIME_V2_CHOP_EDGE_MULT: float = 1.10
    STRATEGY_REGIME_V2_COUNTERTREND_EDGE_MULT: float = 1.25
    STRATEGY_REGIME_V2_TOXIC_SOFT_OFFSET_ADD_MAX_TICKS: int = 4
    STRATEGY_REGIME_V2_CHOP_OFFSET_ADD_TICKS: int = 1
    STRATEGY_REGIME_V2_COUNTERTREND_OFFSET_ADD_TICKS: int = 1

    # Fill-quality feedback loop (uses fq_{H}_mean_ticks_buy/sell from Store stats)
    STRATEGY_FQ_HORIZON_MS: int = 500
    STRATEGY_FQ_MIN_N: int = 30
    STRATEGY_FQ_EDGE_SCALE_TICKS: float = 0.5
    STRATEGY_FQ_NEG_ENTRY_MULT_MAX: float = 2.0
    STRATEGY_FQ_NEG_QUOTE_INTERVAL_MAX_MS: int = 600

    # --- Level 3: microstructure (inventory control + 2-sided quoting + smart requote) ---
    STRATEGY_MM_ENABLED: bool = True
    # Base inside ticks for both sides; defaults to STRATEGY_PEG_INSIDE_TICKS.
    STRATEGY_MM_BASE_INSIDE_TICKS: int = 0
    STRATEGY_MM_MAX_INSIDE_TICKS: int = 3
    # Skew how aggressive we quote based on signal (imbalance/pressure)
    STRATEGY_MM_SIGNAL_SKEW_TICKS: int = 2
    # Inventory skew: when in position, reduce adding-side aggressiveness and ease exit-side.
    STRATEGY_MM_INV_SKEW_TICKS: int = 2
    STRATEGY_MM_INV_MAX_POS_QTY: float = 0.003
    STRATEGY_MM_INV_ADD_QTY_MULT: float = 0.50
    # Use fill-quality feedback to widen (less aggressive) quotes per-side
    STRATEGY_MM_FQ_WIDEN_TICKS: int = 2
    # Sticky quotes: avoid chasing every tick; only chase when far, otherwise keep resting orders.
    STRATEGY_MM_STICKY_QUOTES: bool = True

    # Smart requote controls
    # Requote controls (Level-3): keep these conservative to avoid churn/cancel spam.
    STRATEGY_MM_REQUOTE_PRICE_TICKS: int = 8
    STRATEGY_MM_REQUOTE_MIN_INTERVAL_MS: int = 2500
    # Anti-flap: do not cancel/replace very fresh orders on small drifts.
    STRATEGY_MM_MIN_ORDER_LIFE_MS: int = 2500

    # --- Level 4: micro-ML + bandit (no LLM) ---
    STRATEGY_ML_ENABLED: bool = True
    STRATEGY_ML_PREFER_H_MS: int = 500
    STRATEGY_ML_WARMUP_N: int = 200
    STRATEGY_ML_MIN_EDGE: float = 0.06
    STRATEGY_ML_ONE_SIDED_EDGE: float = 0.12
    STRATEGY_ML_SKEW_TICKS: int = 2
    STRATEGY_ML_PROBE_INTERVAL_MS: int = 120
    STRATEGY_ML_LR: float = 0.05
    STRATEGY_ML_L2: float = 0.0001

    STRATEGY_BANDIT_ENABLED: bool = True
    STRATEGY_BANDIT_SWITCH_MS: int = 30000
    STRATEGY_BANDIT_EPSILON: float = 0.15
    STRATEGY_BANDIT_EPS_MIN: float = 0.03
    STRATEGY_BANDIT_EPS_DECAY: float = 0.995
    STRATEGY_BANDIT_EWMA_ALPHA: float = 0.25

    # Matching engine tuning (paper-only)
    #  - queue factor < 1.0 means we assume we join not at the absolute back of the visible queue.
    QUEUE_AHEAD_FACTOR: float = 0.5
    #  - when aggTrade flow is sparse (common on testnet), infer consumption from top-of-book size reductions.
    SYNTH_FILL_FROM_BOOK: bool = True
    SYNTH_MIN_TRADE_QUIET_MS: int = 250

    # risk
    # Testnet market data can be bursty; allow more slack before killing.
    MAX_BOOK_STALE_MS: int = 15000
    MAX_EVENT_LAG_MS: int = 15000
    # Absolute *sanity* cap on position quantity. Real risk cap is enforced via
    # a notional-based limit in RiskManager (equity * STRATEGY_MAX_NOTIONAL_PCT).
    # Keep this high to avoid false kill-switch triggers across symbols.
    MAX_POSITION_QTY: float = 1000.0

    # Allow some headroom above the strategy notional cap before triggering kill.
    # This prevents false kills from small price moves / mark-mid spread.
    RISK_MAX_POS_NOTIONAL_MULT: float = 1.25
    # Level-3 market making uses 2 open orders (bid+ask) when allowed.
    MAX_OPEN_ORDERS: int = 2
    # Protective limiter against runaway order loops.
    # For Level-3 two-sided quoting this must be higher than the Level-1/2 defaults,
    # otherwise normal requotes will trip the kill-switch.
    MAX_ORDERS_PER_MIN: int = 120
    MAX_DAILY_LOSS_USDT: float = 200.0

    # dashboard
    ROLLING_POINTS: int = 600
    DASH_PUSH_MS: int = 250

    def __init__(self, **data):
        # hard-set WS base to market data testnet base required by spec
        data.setdefault("WS_BASE", "wss://fstream.binancefuture.com")
        super().__init__(**data)

        # enforce testnet-only endpoints
        if "demo-fapi.binance.com" not in self.REST_BASE:
            raise ValueError("REST_BASE must be https://demo-fapi.binance.com (testnet only)")
        if self.WS_BASE != "wss://fstream.binancefuture.com":
            raise ValueError("WS_BASE must be wss://fstream.binancefuture.com (market data testnet base)")

