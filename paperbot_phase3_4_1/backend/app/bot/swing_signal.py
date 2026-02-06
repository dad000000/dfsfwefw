from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional, Tuple


def _clamp(x: float, lo: float, hi: float) -> float:
    return float(max(float(lo), min(float(hi), float(x))))


def _ema_update(prev: Optional[float], x: float, alpha: float) -> float:
    if prev is None or not math.isfinite(prev):
        return float(x)
    return float(prev + alpha * (float(x) - float(prev)))


@dataclass
class SwingPrediction:
    dir: int              # -1, 0, +1
    strength: float       # 0..1
    edge_ticks_est: float # expected move in ticks (1-5 min horizon)
    ret_5s: float
    ret_30s: float
    ret_60s: float
    vol_abs_ret_ema: float
    spread_ticks: float
    imbalance_ema: float
    pressure_ema: float
    churn_ema: float
    score_ticks: float
    # Phase 3.2 features
    atr_ticks_120: float
    slope_norm: float
    imb_std: float
    mark_last_div_ticks: float


class SwingSignal:
    """Lightweight 1–5 minute swing signal (no ML, EMA features).

    Observes market microstructure and recent returns and emits:
      - direction (dir) in {-1,0,+1}
      - strength in [0..1]
      - edge estimate in ticks for 60–300s horizon

    It also enforces hysteresis / min-dwell to avoid flipping every second.
    """

    def __init__(
        self,
        *,
        max_hist_ms: int = 300_000,
        min_dwell_ms: int = 10_000,
        flip_stronger_mult: float = 1.5,
    ) -> None:
        self._hist: Deque[Tuple[int, float]] = deque()
        self._max_hist_ms = int(max_hist_ms)
        self._min_dwell_ms = int(min_dwell_ms)
        self._flip_stronger_mult = float(flip_stronger_mult)

        self._last_ts_ms: int = 0
        self._last_mid: float = 0.0

        # EMA features
        self.ret_5s_ema: Optional[float] = None
        self.ret_30s_ema: Optional[float] = None
        self.ret_60s_ema: Optional[float] = None
        self.vol_abs_ret_ema: Optional[float] = None  # EMA(abs return) ~60s horizon
        self.spread_ticks_ema: Optional[float] = None
        self.imbalance_ema: Optional[float] = None
        self.pressure_ema: Optional[float] = None
        self.churn_ema: Optional[float] = None

        # Phase 3.2: ATR proxy (ticks) + slope + imbalance variance + mark/last divergence
        self.atr_ticks_120_ema: Optional[float] = None
        self.mid_60s_ema: Optional[float] = None
        self.mid_300s_ema: Optional[float] = None
        self.imbalance2_ema: Optional[float] = None
        self.mark_last_div_ticks: float = 0.0

        # hysteresis state
        self._dir: int = 0
        self._dir_ts_ms: int = 0

    def _trim(self, now_ms: int) -> None:
        cutoff = int(now_ms) - int(self._max_hist_ms) - 1000
        while self._hist and int(self._hist[0][0]) < cutoff:
            self._hist.popleft()

    def _return_over(self, now_ms: int, horizon_ms: int, mid: float) -> float:
        """Return (fraction) over horizon, using best available point <= horizon ago."""
        if mid <= 0:
            return 0.0
        target = int(now_ms) - int(horizon_ms)
        prev_px = None
        # walk from right to left (latest -> older) until we cross target
        for ts, px in reversed(self._hist):
            if int(ts) <= target:
                prev_px = float(px)
                break
        if prev_px is None or prev_px <= 0:
            return 0.0
        return float((float(mid) - float(prev_px)) / float(prev_px))

    def observe(
        self,
        ts_ms: int,
        *,
        mid: float,
        tick_size: float,
        spread: float,
        imbalance: float,
        trade_pressure: float,
        churn_proxy: float,
        mark_price: Optional[float] = None,
        last_trade_price: Optional[float] = None,
    ) -> None:
        ts_ms = int(ts_ms)
        mid = float(mid)
        tick = float(tick_size)
        spread = float(spread)
        imb = float(imbalance)
        press = float(trade_pressure)
        churn = float(churn_proxy)

        if ts_ms <= 0 or mid <= 0 or tick <= 0:
            return

        if self._last_ts_ms <= 0:
            self._last_ts_ms = ts_ms
            self._last_mid = mid
            self._hist.append((ts_ms, mid))
            self._dir_ts_ms = ts_ms
            return

        dt_ms = max(1, int(ts_ms) - int(self._last_ts_ms))
        self._last_ts_ms = ts_ms
        self._last_mid = mid

        self._hist.append((ts_ms, mid))
        self._trim(ts_ms)

        # raw returns
        r5 = self._return_over(ts_ms, 5_000, mid)
        r30 = self._return_over(ts_ms, 30_000, mid)
        r60 = self._return_over(ts_ms, 60_000, mid)

        # EMA alphas with time constant approx equal to the horizon
        a5 = 1.0 - math.exp(-float(dt_ms) / 5_000.0)
        a30 = 1.0 - math.exp(-float(dt_ms) / 30_000.0)
        a60 = 1.0 - math.exp(-float(dt_ms) / 60_000.0)
        a_vol = 1.0 - math.exp(-float(dt_ms) / 60_000.0)
        a_sp = 1.0 - math.exp(-float(dt_ms) / 10_000.0)
        a_imb = 1.0 - math.exp(-float(dt_ms) / 15_000.0)
        a_press = 1.0 - math.exp(-float(dt_ms) / 5_000.0)
        a_churn = 1.0 - math.exp(-float(dt_ms) / 30_000.0)

        self.ret_5s_ema = _ema_update(self.ret_5s_ema, r5, a5)
        self.ret_30s_ema = _ema_update(self.ret_30s_ema, r30, a30)
        self.ret_60s_ema = _ema_update(self.ret_60s_ema, r60, a60)

        # vol proxy: abs(instant return)
        inst_ret = (mid - float(self._hist[-2][1])) / max(float(self._hist[-2][1]), 1e-12) if len(self._hist) >= 2 else 0.0
        self.vol_abs_ret_ema = _ema_update(self.vol_abs_ret_ema, abs(float(inst_ret)), a_vol)

        # Phase 3.2: ATR-like proxy in ticks (EMA(abs mid_ret_ticks)) ~120s
        inst_ret_ticks = ((mid - float(self._hist[-2][1])) / float(tick)) if len(self._hist) >= 2 else 0.0
        a_atr120 = 1.0 - math.exp(-float(dt_ms) / 120_000.0)
        self.atr_ticks_120_ema = _ema_update(self.atr_ticks_120_ema, abs(float(inst_ret_ticks)), a_atr120)

        # Phase 3.2: slope proxy via EMA(mid,60s) - EMA(mid,300s)
        a_mid60 = 1.0 - math.exp(-float(dt_ms) / 60_000.0)
        a_mid300 = 1.0 - math.exp(-float(dt_ms) / 300_000.0)
        self.mid_60s_ema = _ema_update(self.mid_60s_ema, float(mid), a_mid60)
        self.mid_300s_ema = _ema_update(self.mid_300s_ema, float(mid), a_mid300)

        sp_ticks = float(spread) / float(tick) if tick > 0 else 0.0
        self.spread_ticks_ema = _ema_update(self.spread_ticks_ema, sp_ticks, a_sp)

        self.imbalance_ema = _ema_update(self.imbalance_ema, _clamp(imb, -1.0, 1.0), a_imb)
        self.imbalance2_ema = _ema_update(self.imbalance2_ema, float(_clamp(imb, -1.0, 1.0)) ** 2, a_imb)
        self.pressure_ema = _ema_update(self.pressure_ema, _clamp(press, -1.0, 1.0), a_press)
        self.churn_ema = _ema_update(self.churn_ema, max(0.0, churn), a_churn)

        # Phase 3.2: mark/last divergence in ticks (instant, for diagnostics)
        try:
            mp = float(mark_price) if mark_price is not None else 0.0
            lp = float(last_trade_price) if last_trade_price is not None else 0.0
            if tick > 0 and mp > 0 and lp > 0:
                self.mark_last_div_ticks = float((mp - lp) / float(tick))
            else:
                self.mark_last_div_ticks = 0.0
        except Exception:
            self.mark_last_div_ticks = 0.0

    def predict(self, now_ms: int, *, mid: float, tick_size: float) -> SwingPrediction:
        """Produce a stable swing prediction."""
        now_ms = int(now_ms)
        mid = float(mid)
        tick = float(tick_size)

        r5 = float(self.ret_5s_ema or 0.0)
        r30 = float(self.ret_30s_ema or 0.0)
        r60 = float(self.ret_60s_ema or 0.0)

        imb = float(self.imbalance_ema or 0.0)
        press = float(self.pressure_ema or 0.0)
        sp_ticks = float(self.spread_ticks_ema or 0.0)
        churn = float(self.churn_ema or 0.0)
        vol_abs = float(self.vol_abs_ret_ema or 0.0)

        # Convert returns to ticks for a 1–5 min horizon proxy
        def to_ticks(ret_frac: float) -> float:
            if mid <= 0 or tick <= 0:
                return 0.0
            return float(ret_frac) * float(mid) / float(tick)

        r5_t = to_ticks(r5)
        r30_t = to_ticks(r30)
        r60_t = to_ticks(r60)

        # Momentum-ish score with light microstructure bias.
        #  - emphasize 30s/60s (swing horizon), keep 5s as a confirmation.
        # NOTE: Microstructure is noisy on testnet (and can be stale during book resync).
        # We keep micro bias small and include *trade pressure* as a confirmation so we
        # don't end up repeatedly leaning against the tape when L2 imbalance is misleading.
        micro_scale = max(1.0, min(10.0, sp_ticks))
        micro_bias = (0.08 * imb + 0.18 * press) * micro_scale
        score_ticks = (
            0.10 * r5_t
            + 0.55 * r30_t
            + 0.85 * r60_t
            + micro_bias
        )

        # Penalize very churny regimes a bit (to avoid overtrading noise)
        score_ticks *= float(1.0 / (1.0 + 0.08 * max(0.0, churn)))

        # Dynamic neutral band: at least a couple ticks, grows with spread.
        base_th = max(3.0, 0.8 * sp_ticks)
        dir_raw = 0
        if score_ticks > base_th:
            dir_raw = +1
        elif score_ticks < -base_th:
            dir_raw = -1

        # Strength: smooth saturating function of score magnitude.
        # Scale by typical swing move size (in ticks): allow strength to grow by ~30 ticks.
        k = max(10.0, 25.0 + 0.5 * sp_ticks)
        strength = float(math.tanh(abs(score_ticks) / k))
        strength = _clamp(strength, 0.0, 1.0)

        # Edge estimate: blend score magnitude and volatility, cap for stability.
        vol_ticks = float(vol_abs) * float(mid) / float(tick) if mid > 0 and tick > 0 else 0.0
        atr_ticks_120 = float(self.atr_ticks_120_ema or 0.0)
        # Prefer tick-based ATR proxy when available
        vol_ticks_eff = float(atr_ticks_120) if float(atr_ticks_120) > 0 else float(vol_ticks)

        # Phase 3.2 slope proxy (normalized)
        slope_ticks = 0.0
        try:
            if tick > 0:
                slope_ticks = (float(self.mid_60s_ema or mid) - float(self.mid_300s_ema or mid)) / float(tick)
        except Exception:
            slope_ticks = 0.0
        slope_norm = float(slope_ticks) / max(float(atr_ticks_120), 1e-6)

        # Imbalance std (EMA-based)
        imb2 = float(self.imbalance2_ema or 0.0)
        imb_var = max(0.0, float(imb2) - float(imb) * float(imb))
        imb_std = float(math.sqrt(imb_var))

        mark_last_div_ticks = float(getattr(self, "mark_last_div_ticks", 0.0) or 0.0)
        edge = abs(score_ticks) * 0.9 + vol_ticks_eff * 0.6
        edge = _clamp(edge, 0.0, 200.0)

        # Hysteresis / min-dwell
        dir_out = dir_raw
        since_flip = int(now_ms) - int(self._dir_ts_ms or 0)
        if dir_raw != self._dir:
            if since_flip < int(self._min_dwell_ms):
                dir_out = int(self._dir)
            else:
                # require stronger evidence to flip
                th_flip = float(base_th) * float(self._flip_stronger_mult)
                if abs(score_ticks) < th_flip:
                    dir_out = int(self._dir)
                else:
                    self._dir = int(dir_raw)
                    self._dir_ts_ms = int(now_ms)
                    dir_out = int(dir_raw)
        else:
            # keep timestamp fresh if we're confidently in a direction
            if dir_out != 0:
                self._dir_ts_ms = int(self._dir_ts_ms or now_ms)

        return SwingPrediction(
            dir=int(dir_out),
            strength=float(strength),
            edge_ticks_est=float(edge),
            ret_5s=float(r5),
            ret_30s=float(r30),
            ret_60s=float(r60),
            vol_abs_ret_ema=float(vol_abs),
            spread_ticks=float(sp_ticks),
            imbalance_ema=float(imb),
            pressure_ema=float(press),
            churn_ema=float(churn),
            score_ticks=float(score_ticks),
            atr_ticks_120=float(atr_ticks_120),
            slope_norm=float(slope_norm),
            imb_std=float(imb_std),
            mark_last_div_ticks=float(mark_last_div_ticks),
        )

