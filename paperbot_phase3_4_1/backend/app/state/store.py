from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Tuple, Optional

from ..config import Settings
from ..utils.time_ms import now_ms
from .models import (
    AccountState,
    ClosedTrade,
    Fill,
    L2Book,
    MarkPrice,
    Order,
    TradePrint,
)

log = logging.getLogger("state.store")


@dataclass
class RollingSeries:
    ts_ms: Deque[int] = field(default_factory=deque)
    equity: Deque[float] = field(default_factory=deque)
    realized: Deque[float] = field(default_factory=deque)
    unrealized: Deque[float] = field(default_factory=deque)

    def append(self, ts_ms: int, equity: float, realized: float, unrealized: float, max_points: int) -> None:
        self.ts_ms.append(ts_ms)
        self.equity.append(equity)
        self.realized.append(realized)
        self.unrealized.append(unrealized)
        while len(self.ts_ms) > max_points:
            self.ts_ms.popleft()
            self.equity.popleft()
            self.realized.popleft()
            self.unrealized.popleft()


@dataclass
class FillProbe:
    """Tracks post-fill price movement (in ticks) at fixed horizons.

    Positive outcome means the mid moved in our favor:
      - BUY: mid_up => positive
      - SELL: mid_down => positive
    """

    ts_ms: int
    side: str  # BUY/SELL
    entry_mid: float
    horizons_ms: Tuple[int, ...] = (250, 500, 1000)
    done: Dict[int, float] = field(default_factory=dict)  # horizon_ms -> outcome_ticks


class Store:
    def __init__(self, settings: Settings) -> None:
        self.s = settings
        self.lock = asyncio.Lock()

        self.book = L2Book()
        self.trades: Deque[TradePrint] = deque(maxlen=200)
        self.mark = MarkPrice(ts_ms=0, mark_price=0.0)

        self.orders: Dict[str, Order] = {}
        self.fills: Deque[Fill] = deque(maxlen=500)
        # Keep more than the dashboard window so offline backtests (Phase 2.2)
        # can export a full trade history. Dashboard still slices to 50.
        self.closed_trades: Deque[ClosedTrade] = deque(maxlen=5000)

        self.account = AccountState(
            cash_balance_usdt=self.s.PAPER_START_BALANCE_USDT,
            position_qty=0.0,
            avg_entry_price=0.0,
            realized_pnl_usdt=0.0,
            unrealized_pnl_usdt=0.0,
            fees_paid_usdt=0.0,
            equity_usdt=self.s.PAPER_START_BALANCE_USDT,
            notional_usdt=0.0,
            margin_used_usdt=0.0,
        )

        self.series = RollingSeries()

        # control + status
        self.bot_running: bool = False
        self.kill_switch: bool = False
        self.kill_reason: str = ""
        self.recent_rejects: Deque[int] = deque(maxlen=200)  # timestamps
        self.order_rate: Deque[int] = deque(maxlen=200)      # order timestamps

        # computed market metrics
        self.tick_size: float = 0.1
        self.microprice: float = 0.0
        self.mid: float = 0.0
        self.spread: float = 0.0
        self.imbalance: float = 0.0
        self.volatility: float = 0.0
        self._mark_hist: Deque[float] = deque(maxlen=200)

        # trade pressure window (last ~2s)
        self._buy_vol: Deque[Tuple[int, float]] = deque()
        self._sell_vol: Deque[Tuple[int, float]] = deque()

        # ws connectivity (set from ws client)
        self.ws_connected: bool = False
        self.ws_reconnects: int = 0
        self.ws_last_event_ts_ms: int = 0

        # strategy meta (for dashboard observability)
        self.strategy_meta: Dict[str, object] = {}

        # KPI rolling windows (Phase 1: observability)
        # Keep these lightweight; swing trades per hour are small.
        # We keep at most ~1h worth of fills and ~1d worth of closed trades.
        self._fee_events: Deque[Tuple[int, float]] = deque(maxlen=20_000)        # (ts_ms, fee_usdt)
        self._turnover_events: Deque[Tuple[int, float]] = deque(maxlen=20_000)   # (ts_ms, notional_usdt)
        self._closed_trade_events: Deque[Tuple[int, int]] = deque(maxlen=10_000) # (exit_ts_ms, hold_ms)
        self._turnover_total_usdt: float = 0.0


        # execution diagnostics (Level 1)
        self.exec_fills_total: int = 0
        self.exec_cancels_total: int = 0
        self.exec_rejects_total: int = 0

        self._fill_ts_ms: Deque[int] = deque(maxlen=2000)
        self._cancel_ts_ms: Deque[int] = deque(maxlen=2000)
        self._reject_ts_ms: Deque[int] = deque(maxlen=2000)

        # Post-fill outcome probes (quality of fills)
        self._fill_probes: Deque[FillProbe] = deque(maxlen=400)
        self._fq_all: Dict[int, Deque[float]] = {250: deque(maxlen=400), 500: deque(maxlen=400), 1000: deque(maxlen=400)}
        self._fq_buy: Dict[int, Deque[float]] = {250: deque(maxlen=400), 500: deque(maxlen=400), 1000: deque(maxlen=400)}
        self._fq_sell: Dict[int, Deque[float]] = {250: deque(maxlen=400), 500: deque(maxlen=400), 1000: deque(maxlen=400)}

    def _now_ms(self) -> int:
        return int(now_ms())

    async def set_tick_size(self, tick: float) -> None:
        async with self.lock:
            self.tick_size = max(float(tick), 1e-9)

    async def update_ws_status(self, connected: bool, reconnects: int, last_event_ts_ms: int) -> None:
        async with self.lock:
            self.ws_connected = bool(connected)
            self.ws_reconnects = int(reconnects)
            self.ws_last_event_ts_ms = int(last_event_ts_ms)

    async def on_mark(self, ts_ms: int, mark_price: float) -> None:
        async with self.lock:
            self.mark = MarkPrice(ts_ms=int(ts_ms), mark_price=float(mark_price))
            self._mark_hist.append(float(mark_price))

            # simple volatility proxy: std dev of returns (last ~60 marks)
            if len(self._mark_hist) >= 20:
                vals = list(self._mark_hist)[-60:]
                rets: List[float] = []
                for i in range(1, len(vals)):
                    prev = vals[i - 1]
                    if prev > 0:
                        rets.append((vals[i] - prev) / prev)
                if len(rets) >= 10:
                    mean = sum(rets) / len(rets)
                    var = sum((r - mean) ** 2 for r in rets) / max(len(rets) - 1, 1)
                    self.volatility = var ** 0.5

            # Update post-fill probes using wall-clock time.
            # (Exchange event timestamps can drift vs local clock on some setups.)
            self._update_fill_probes_locked(self._now_ms())

            await self._recompute_account_locked()
            self._append_series_locked()

    async def on_trade_print(self, t: TradePrint) -> None:
        async with self.lock:
            self.trades.appendleft(t)
            now_ms = int(t.ts_ms)
            if t.side == "BUY":
                self._buy_vol.append((now_ms, float(t.qty)))
            else:
                self._sell_vol.append((now_ms, float(t.qty)))
            self._trim_pressure_locked(now_ms)

    def _trim_pressure_locked(self, now_ms: int) -> None:
        cutoff = now_ms - 2000
        while self._buy_vol and self._buy_vol[0][0] < cutoff:
            self._buy_vol.popleft()
        while self._sell_vol and self._sell_vol[0][0] < cutoff:
            self._sell_vol.popleft()

    def trade_pressure_locked(self) -> float:
        buy = sum(q for _, q in self._buy_vol)
        sell = sum(q for _, q in self._sell_vol)
        denom = buy + sell
        if denom <= 0:
            return 0.0
        return (buy - sell) / denom

    async def set_book_from_snapshot(
        self,
        last_update_id: int,
        bids: List[Tuple[float, float]],
        asks: List[Tuple[float, float]],
    ) -> None:
        async with self.lock:
            self.book.bids = {float(p): float(q) for p, q in bids if float(q) > 0}
            self.book.asks = {float(p): float(q) for p, q in asks if float(q) > 0}
            self.book.last_update_id = int(last_update_id)
            now_ms = int(self._now_ms())
            self.book.last_event_ts_ms = now_ms
            self.book.last_exchange_ts_ms = 0
            self.book.resyncs += 1

            # important: after snapshot we have NOT started strict delta sequencing yet
            self.book.depth_seq_started = False

            self._recompute_market_metrics_locked()
            self._update_fill_probes_locked(self._now_ms())

    async def apply_depth_delta(
        self,
        event_ts_ms: int,
        first_update_id: int,
        final_update_id: int,
        bid_updates: List[Tuple[float, float]],
        ask_updates: List[Tuple[float, float]],
    ) -> bool:
        """
        Correct Binance diff depth stitching:
          - discard if u <= lastUpdateId
          - first applicable delta after snapshot must satisfy: U <= lastUpdateId+1 <= u
          - after first applied delta: require strict U == lastUpdateId+1
        """
        async with self.lock:
            last_id = int(self.book.last_update_id)
            if last_id == 0:
                return False

            first_update_id = int(first_update_id)
            final_update_id = int(final_update_id)

            if final_update_id <= last_id:
                return True  # old / already applied

            # Must cover last_id+1
            if not (first_update_id <= last_id + 1 <= final_update_id):
                return False

            # After sequencing has started, enforce strict contiguity
            if self.book.depth_seq_started and first_update_id != last_id + 1:
                return False

            for p, q in bid_updates:
                fp = float(p)
                fq = float(q)
                if fq <= 0:
                    self.book.bids.pop(fp, None)
                else:
                    self.book.bids[fp] = fq

            for p, q in ask_updates:
                fp = float(p)
                fq = float(q)
                if fq <= 0:
                    self.book.asks.pop(fp, None)
                else:
                    self.book.asks[fp] = fq

            self.book.last_update_id = final_update_id
            self.book.last_exchange_ts_ms = int(event_ts_ms)
            # Use local receipt time for staleness/age checks (avoid exchange vs local clock drift)
            self.book.last_event_ts_ms = int(self._now_ms())
            self.book.depth_seq_started = True

            self._recompute_market_metrics_locked()
            self._update_fill_probes_locked(self._now_ms())
            return True


    async def set_top_of_book(
        self,
        ts_ms: int,
        best_bid: float,
        best_bid_qty: float,
        best_ask: float,
        best_ask_qty: float,
        *,
        tick_size: Optional[float] = None,
    ) -> None:
        """Offline helper (Phase 2.2): set a 1-level book (top-of-book only).

        This intentionally bypasses Binance snapshot+delta sequencing and is ONLY
        meant for offline replay/backtests that consume recorded top-of-book.
        """
        async with self.lock:
            ts_ms = int(ts_ms)
            if tick_size is not None:
                try:
                    ft = float(tick_size)
                    if ft > 0:
                        self.tick_size = ft
                except Exception:
                    pass

            bb = float(best_bid)
            bq = float(best_bid_qty)
            ba = float(best_ask)
            aq = float(best_ask_qty)

            self.book.bids = {bb: max(0.0, bq)} if (bb > 0 and bq > 0) else {}
            self.book.asks = {ba: max(0.0, aq)} if (ba > 0 and aq > 0) else {}
            self.book.last_event_ts_ms = ts_ms
            self.book.last_exchange_ts_ms = ts_ms

            # Do NOT touch last_update_id / depth_seq_started / resyncs.
            self._recompute_market_metrics_locked()
            self._update_fill_probes_locked(self._now_ms())

    def _recompute_market_metrics_locked(self) -> None:
        bb, ba = self.book.best_bid_ask()
        if bb is None or ba is None:
            self.mid = 0.0
            self.spread = 0.0
            self.microprice = 0.0
            self.imbalance = 0.0
            return

        self.mid = (bb + ba) / 2.0
        self.spread = ba - bb

        bid_qty = float(self.book.bids.get(bb, 0.0))
        ask_qty = float(self.book.asks.get(ba, 0.0))
        denom = bid_qty + ask_qty
        if denom > 0:
            self.microprice = (bb * ask_qty + ba * bid_qty) / denom
        else:
            self.microprice = self.mid

        K = 5
        bid_lvls, ask_lvls = self.book.top_levels(K)
        bsum = sum(float(q) for _, q in bid_lvls)
        asum = sum(float(q) for _, q in ask_lvls)
        den = bsum + asum
        self.imbalance = (bsum - asum) / den if den > 0 else 0.0

    async def set_bot_running(self, running: bool) -> None:
        async with self.lock:
            self.bot_running = bool(running)

    async def trigger_kill(self, reason: str) -> None:
        async with self.lock:
            self.kill_switch = True
            self.kill_reason = str(reason)
            self.bot_running = False

    async def clear_kill(self) -> None:
        async with self.lock:
            self.kill_switch = False
            self.kill_reason = ""

    async def reset_session(self) -> None:
        async with self.lock:
            self.book = L2Book()
            self.trades.clear()
            self.mark = MarkPrice(ts_ms=0, mark_price=0.0)
            self.orders.clear()
            self.fills.clear()
            self.closed_trades.clear()
            self.account = AccountState(
                cash_balance_usdt=self.s.PAPER_START_BALANCE_USDT,
                position_qty=0.0,
                avg_entry_price=0.0,
                realized_pnl_usdt=0.0,
                unrealized_pnl_usdt=0.0,
                fees_paid_usdt=0.0,
                equity_usdt=self.s.PAPER_START_BALANCE_USDT,
                notional_usdt=0.0,
                margin_used_usdt=0.0,
            )
            self.series = RollingSeries()
            self.kill_switch = False
            self.kill_reason = ""
            self.bot_running = False
            self.recent_rejects.clear()
            self.order_rate.clear()
            self.microprice = self.mid = self.spread = self.imbalance = self.volatility = 0.0
            self._mark_hist.clear()
            self._buy_vol.clear()
            self._sell_vol.clear()
            self._fee_events.clear()
            self._turnover_events.clear()
            self._closed_trade_events.clear()
            self._turnover_total_usdt = 0.0
            self.exec_fills_total = 0
            self.exec_cancels_total = 0
            self.exec_rejects_total = 0
            self._fill_ts_ms.clear()
            self._cancel_ts_ms.clear()
            self._reject_ts_ms.clear()
            self._fill_probes.clear()
            for k in self._fq_all: self._fq_all[k].clear()
            for k in self._fq_buy: self._fq_buy[k].clear()
            for k in self._fq_sell: self._fq_sell[k].clear()

    async def _recompute_account_locked(self) -> None:
        mp = float(self.mark.mark_price)
        qty = float(self.account.position_qty)
        avg = float(self.account.avg_entry_price)

        upnl = qty * (mp - avg) if (avg > 0 and mp > 0) else 0.0
        notional = abs(qty) * mp
        margin = notional / max(float(self.s.LEVERAGE), 1e-9)
        equity = float(self.account.cash_balance_usdt) + upnl

        self.account = self.account.model_copy(update={
            "unrealized_pnl_usdt": upnl,
            "equity_usdt": equity,
            "notional_usdt": notional,
            "margin_used_usdt": margin,
        })

    def _append_series_locked(self) -> None:
        now_ms = self._now_ms()
        self.series.append(
            ts_ms=now_ms,
            equity=float(self.account.equity_usdt),
            realized=float(self.account.realized_pnl_usdt),
            unrealized=float(self.account.unrealized_pnl_usdt),
            max_points=int(self.s.ROLLING_POINTS),
        )

    def on_paper_fill_locked(self, side: str, ts_ms: int) -> None:
        """Record a paper fill. store.lock MUST be held."""
        self.exec_fills_total += 1
        self._fill_ts_ms.append(int(ts_ms))

        # Record a post-fill probe using current mid (fallback to mark)
        mid = float(self.mid) if float(self.mid) > 0 else float(self.mark.mark_price)
        if mid > 0:
            self._fill_probes.append(FillProbe(ts_ms=int(ts_ms), side=str(side), entry_mid=mid))

    def on_paper_fill_details_locked(self, f: Fill) -> None:
        """Record fill-derived KPI events (fees/turnover). store.lock MUST be held."""
        try:
            ts = int(getattr(f, "ts_ms", 0) or 0)
            fee = float(getattr(f, "fee_usdt", 0.0) or 0.0)
            notional = float(getattr(f, "price", 0.0) or 0.0) * float(getattr(f, "qty", 0.0) or 0.0)
        except Exception:
            return

        self._turnover_total_usdt += max(0.0, float(notional))
        self._fee_events.append((ts, float(fee)))
        self._turnover_events.append((ts, float(notional)))

        # Trim to rolling windows: keep ~1h of fills for fees/turnover.
        cutoff_1h = ts - 3_700_000  # 1h + small buffer
        while self._fee_events and int(self._fee_events[0][0]) < int(cutoff_1h):
            self._fee_events.popleft()
        while self._turnover_events and int(self._turnover_events[0][0]) < int(cutoff_1h):
            self._turnover_events.popleft()

    def on_paper_closed_trade_locked(self, ct: ClosedTrade) -> None:
        """Record closed-trade cadence/hold KPIs. store.lock MUST be held."""
        try:
            exit_ts = int(getattr(ct, "exit_ts_ms", 0) or 0)
            hold_ms = int(getattr(ct, "holding_ms", 0) or (int(getattr(ct, "exit_ts_ms", 0) or 0) - int(getattr(ct, "entry_ts_ms", 0) or 0)))
        except Exception:
            return

        self._closed_trade_events.append((exit_ts, max(0, int(hold_ms))))
        # Trim to ~1d
        cutoff_1d = exit_ts - 86_500_000  # 1d + small buffer
        while self._closed_trade_events and int(self._closed_trade_events[0][0]) < int(cutoff_1d):
            self._closed_trade_events.popleft()

    def on_paper_cancel_locked(self, side: str, ts_ms: int) -> None:
        """Record a paper cancel. store.lock MUST be held."""
        self.exec_cancels_total += 1
        self._cancel_ts_ms.append(int(ts_ms))

    def on_paper_reject_locked(self, ts_ms: int) -> None:
        """Record a paper reject. store.lock MUST be held."""
        self.exec_rejects_total += 1
        self._reject_ts_ms.append(int(ts_ms))

    def _kpi_stats_locked(self, now_ms: int) -> dict:
        """KPI observability: fees/turnover cadence + cancel/reject rates."""
        now_ms = int(now_ms)
        w5m = now_ms - 300_000
        w1h = now_ms - 3_600_000
        w1d = now_ms - 86_400_000

        # Fees and turnover (fills)
        fees_1h = float(sum(fee for ts, fee in self._fee_events if int(ts) >= int(w1h)))
        fees_5m = float(sum(fee for ts, fee in self._fee_events if int(ts) >= int(w5m)))
        turnover_1h = float(sum(n for ts, n in self._turnover_events if int(ts) >= int(w1h)))
        turnover_5m = float(sum(n for ts, n in self._turnover_events if int(ts) >= int(w5m)))

        # Trade cadence (closed trades)
        trades_1h = int(sum(1 for ts, _ in self._closed_trade_events if int(ts) >= int(w1h)))
        trades_1d = int(sum(1 for ts, _ in self._closed_trade_events if int(ts) >= int(w1d)))
        holds_1d = [int(h) for ts, h in self._closed_trade_events if int(ts) >= int(w1d)]
        avg_hold_ms_1d = float(sum(holds_1d) / len(holds_1d)) if holds_1d else 0.0

        # Cancel/reject rates (5m)
        fills_5m = int(sum(1 for t in self._fill_ts_ms if int(t) >= int(w5m)))
        cancels_5m = int(sum(1 for t in self._cancel_ts_ms if int(t) >= int(w5m)))
        rejects_5m = int(sum(1 for t in self._reject_ts_ms if int(t) >= int(w5m)))
        total_events_5m = max(1, fills_5m + cancels_5m + rejects_5m)
        cancel_rate_5m = float(cancels_5m / total_events_5m)
        reject_rate_5m = float(rejects_5m / total_events_5m)

        return {
            "fees_paid_total": float(self.account.fees_paid_usdt),
            "fees_5m": float(fees_5m),
            "fees_1h": float(fees_1h),
            "turnover_total": float(self._turnover_total_usdt),
            "turnover_5m": float(turnover_5m),
            "turnover_1h": float(turnover_1h),
            "trades_1h": int(trades_1h),
            "trades_1d": int(trades_1d),
            "avg_hold_ms_1d": float(avg_hold_ms_1d),
            "fills_5m": int(fills_5m),
            "cancels_5m": int(cancels_5m),
            "rejects_5m": int(rejects_5m),
            "cancel_rate_5m": float(cancel_rate_5m),
            "reject_rate_5m": float(reject_rate_5m),
        }

    def _update_fill_probes_locked(self, now_ms: int) -> None:
        """Resolve any fill probes whose horizons have elapsed (store.lock MUST be held)."""
        if not self._fill_probes:
            return

        mid_now = float(self.mid) if float(self.mid) > 0 else float(self.mark.mark_price)
        if mid_now <= 0:
            return

        tick = max(float(self.tick_size), 1e-9)
        max_h = 1000

        keep: Deque[FillProbe] = deque(maxlen=400)
        for p in list(self._fill_probes):
            # Drop very old probes
            if int(now_ms) - int(p.ts_ms) > (max_h + 10_000):
                continue

            sign = 1.0 if str(p.side).upper() == "BUY" else -1.0
            for h in p.horizons_ms:
                if h in p.done:
                    continue
                if int(now_ms) >= int(p.ts_ms) + int(h):
                    outcome_ticks = sign * ((mid_now - float(p.entry_mid)) / tick)
                    p.done[int(h)] = float(outcome_ticks)
                    self._fq_all[int(h)].append(float(outcome_ticks))
                    if sign > 0:
                        self._fq_buy[int(h)].append(float(outcome_ticks))
                    else:
                        self._fq_sell[int(h)].append(float(outcome_ticks))

            # Keep if not fully done yet
            if len(p.done) < len(p.horizons_ms):
                keep.append(p)

        self._fill_probes = keep

    def _fill_quality_stats_locked(self) -> dict:
        out: dict = {}
        for h in (250, 500, 1000):
            arr = list(self._fq_all.get(h, []))
            if arr:
                out[f"fq_{h}_mean_ticks"] = sum(arr) / len(arr)
                out[f"fq_{h}_hit_rate"] = sum(1 for x in arr if x > 0) / len(arr)
                out[f"fq_{h}_n"] = len(arr)
            else:
                out[f"fq_{h}_mean_ticks"] = 0.0
                out[f"fq_{h}_hit_rate"] = 0.0
                out[f"fq_{h}_n"] = 0

            b = list(self._fq_buy.get(h, []))
            s = list(self._fq_sell.get(h, []))
            out[f"fq_{h}_mean_ticks_buy"] = (sum(b) / len(b)) if b else 0.0
            out[f"fq_{h}_mean_ticks_sell"] = (sum(s) / len(s)) if s else 0.0

        return out

    def _execution_stats_locked(self, now_ms: int) -> dict:
        cutoff = int(now_ms) - 60_000
        fills_1m = sum(1 for t in self._fill_ts_ms if int(t) >= cutoff)
        cancels_1m = sum(1 for t in self._cancel_ts_ms if int(t) >= cutoff)
        rejects_1m = sum(1 for t in self._reject_ts_ms if int(t) >= cutoff)

        fills_total = int(self.exec_fills_total)
        cancels_total = int(self.exec_cancels_total)
        rejects_total = int(self.exec_rejects_total)

        cfr = (cancels_total / fills_total) if fills_total > 0 else float("inf")

        return {
            "fills_total": fills_total,
            "cancels_total": cancels_total,
            "rejects_total": rejects_total,
            "fills_1m": fills_1m,
            "cancels_1m": cancels_1m,
            "rejects_1m": rejects_1m,
            "cancel_fill_ratio": cfr if cfr != float("inf") else 999999.0,
        }



    async def set_strategy_meta(self, meta: Dict[str, object]) -> None:
        """Set small strategy meta fields to be merged into stats in snapshot.

        Keep values JSON-serializable: numbers/strings/bools.
        """
        async with self.lock:
            # shallow merge
            try:
                self.strategy_meta.update(dict(meta))
            except Exception:
                pass

    async def snapshot(self) -> dict:
        async with self.lock:
            now_ms = self._now_ms()
            # Resolve any outstanding fill-quality probes
            self._update_fill_probes_locked(int(now_ms))
            bb, ba = self.book.best_bid_ask()
            bid_lvls, ask_lvls = self.book.top_levels(10)
            last_trades = [t.model_dump() for t in list(self.trades)[:25]]
            closed_trades = [t.model_dump() for t in list(self.closed_trades)[:50]]

            orders_dump = [
                o.model_dump()
                for o in sorted(self.orders.values(), key=lambda x: x.ts_ms, reverse=True)[:200]
            ]

            ws_age_ms = now_ms - self.ws_last_event_ts_ms if self.ws_last_event_ts_ms else None
            book_age_ms = now_ms - self.book.last_event_ts_ms if self.book.last_event_ts_ms else None

            # Stats block: perf + execution + KPIs
            stats = self._summary_stats_locked(list(self.closed_trades))
            stats.update(self._execution_stats_locked(now_ms))
            stats.update(self._kpi_stats_locked(now_ms))
            stats.update(self._fill_quality_stats_locked())

            # Stable strategy_meta schema (Phase 1.1): keys always exist.
            meta_defaults: Dict[str, object] = {
                "mode": "",
                "symbol": "",
                "fee_ticks": 0,
                "min_profit_ticks": 0,
                "swing_dir": 0,
                "swing_strength": 0.0,
                "swing_edge_ticks": 0.0,
                "no_trade_reason": "",
                "gate_state": "",
                "last_step_ts_ms": 0,
                "loop_dt_ms": 0,
                "heartbeat_ok": False,
                "ws_lag_ms": int(ws_age_ms or 0),

                # --- Phase 2.1: NDJSON recorder (defaults) ---
                "recorder_enabled": False,
                "recorder_path": "",
                "recorder_lines": 0,
                "recorder_bytes": 0,
                "recorder_dropped": 0,
                "recorder_queue_len": 0,
                "recorder_last_write_ts_ms": 0,
                "recorder_rotated_parts": 0,
                "recorder_error": "",

                # --- Phase 1.5: rolling execution metrics + autotune/safety (defaults) ---
                "entry_attempts_5m": 0,
                "entry_fills_5m": 0,
                "fill_rate_5m": 0.0,
                "time_to_fill_p50_ms": 0,
                "time_to_fill_p90_ms": 0,
                "ev_gate_open_rate_5m": 0.0,
                "pnl_1h_after_fees": 0.0,

                "autotune_enabled": False,
                "autotune_last_tune_ms": 0,
                "autotune_action": "",
                "autotune_reason": "",
                "autotune_offset_adj_ticks": 0,
                "autotune_entry_cooldown_ms": 0,
                "autotune_requote_min_ms": 0,
                "min_edge_mult_live": 0.0,

                # regime classifier v2 (Phase 3.1)
                "regime_state": "",
                "regime_trend_strength": 0.0,
                "regime_chop_score": 0.0,
                "regime_toxic_score": 0.0,
                "regime_eff_ratio": 0.0,
                "regime_trend_ticks": 0.0,
                "regime_spread_ticks": 0.0,
                "regime_vol_ticks": 0.0,
                "regime_spread_med": 0.0,
                "regime_vol_med": 0.0,
                "regime_toxic_hard": False,
                "regime_toxic_soft": False,
                "regime_countertrend": False,
                "regime_edge_mult": 1.0,
                "regime_offset_add_ticks": 0,
                "min_edge_mult_effective": 0.0,

                # Phase 3.1b: trailing profit (fee-cover activation)
                "trail_enabled": False,
                "trail_active": False,
                "trail_activate_ticks": 0,
                "trail_peak_ticks": 0,
                "trail_retrace_ticks": 0,
                "trail_trigger_level_ticks": 0,
                "pos_delta_ticks": 0,

                "trading_enabled": True,
                "pause_until_ms": 0,
                "pause_reason": "",

                "inactive_reason": "",
                "inactive_top_factors": [],
                "inactive_suggested_fix": "",
            }
            merged_meta = dict(meta_defaults)
            try:
                merged_meta.update(dict(self.strategy_meta or {}))
            except Exception:
                pass

            # Derive heartbeat if we have a last_step timestamp.
            try:
                last_step = int(merged_meta.get("last_step_ts_ms") or 0)
                bot_running = bool(self.bot_running)
                if last_step > 0 and bot_running:
                    merged_meta["heartbeat_ok"] = bool((int(now_ms) - last_step) <= 10_000)
                elif not bot_running:
                    merged_meta["heartbeat_ok"] = False
            except Exception:
                pass

            stats.update(merged_meta)

            bb_qty = float(bid_lvls[0][1]) if bid_lvls else 0.0
            ba_qty = float(ask_lvls[0][1]) if ask_lvls else 0.0
            spread_ticks = float(self.spread) / float(self.tick_size) if float(self.tick_size) > 0 else 0.0
            return {
                "ts_ms": now_ms,
                "connectivity": {
                    "ws_connected": self.ws_connected,
                    "time_source": "local_monotonic_ms",
                    "ws_reconnects": self.ws_reconnects,
                    "ws_last_event_age_ms": ws_age_ms,
                    "book_last_update_age_ms": book_age_ms,
                    "book_last_exchange_ts_ms": int(getattr(self.book, "last_exchange_ts_ms", 0) or 0),
                    "book_resyncs": self.book.resyncs,
                    "book_resync_failures": self.book.resync_failures,
                    "kill_switch": self.kill_switch,
                    "kill_reason": self.kill_reason,
                    "bot_running": self.bot_running,
                },
                "market": {
                    "best_bid": bb,
                    "best_bid_qty": bb_qty,
                    "best_ask": ba,
                    "best_ask_qty": ba_qty,
                    "mid": self.mid,
                    "spread": self.spread,
                    "spread_ticks": spread_ticks,
                    "microprice": self.microprice,
                    "imbalance": self.imbalance,
                    "volatility": self.volatility,
                    "tick_size": self.tick_size,
                    "bids": bid_lvls,
                    "asks": ask_lvls,
                    "trades": last_trades,
                    "trade_pressure": self.trade_pressure_locked(),
                    "mark_price": self.mark.mark_price,
                    "mark_ts_ms": int(self.mark.ts_ms or 0),
                },
                "account": self.account.model_dump(),
                "charts": {
                    "ts_ms": list(self.series.ts_ms),
                    "equity": list(self.series.equity),
                    "realized": list(self.series.realized),
                    "unrealized": list(self.series.unrealized),
                },
                "orders": orders_dump,
                "closed_trades": closed_trades,
                "stats": stats,
            }

    def _summary_stats_locked(self, trades: List[ClosedTrade]) -> dict:
        """Basic performance summary over closed trades (store.lock MUST be held)."""
        if not trades:
            return {
                "win_rate": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "profit_factor": 0.0,
                "max_drawdown": 0.0,
                "trades_n": 0,
                "wins_n": 0,
                "losses_n": 0,
                "net_pnl_sum": 0.0,
                "net_pnl_long_sum": 0.0,
                "net_pnl_short_sum": 0.0,
                "long_trades_n": 0,
                "short_trades_n": 0,
                "avg_pnl_per_trade": 0.0,
                "avg_hold_ms": 0.0,
            }

        pnls = [float(t.realized_pnl_usdt) - float(t.fees_usdt) for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        win_rate = len(wins) / len(pnls) if pnls else 0.0
        avg_win = sum(wins) / len(wins) if wins else 0.0
        avg_loss = sum(losses) / len(losses) if losses else 0.0
        profit_factor = (
            (sum(wins) / abs(sum(losses)))
            if losses and sum(losses) != 0
            else (float("inf") if wins else 0.0)
        )

        eq = list(self.series.equity)
        max_dd = 0.0
        peak = eq[0] if eq else 0.0
        for x in eq:
            peak = max(peak, x)
            max_dd = max(max_dd, peak - x)

        hold_ms = [int(t.exit_ts_ms) - int(t.entry_ts_ms) for t in trades]
        avg_hold_ms = (sum(hold_ms) / len(hold_ms)) if hold_ms else 0.0

        net = float(sum(pnls))
        long_net = float(sum((float(t.realized_pnl_usdt) - float(t.fees_usdt)) for t in trades if str(t.side).upper() == "LONG"))
        short_net = float(sum((float(t.realized_pnl_usdt) - float(t.fees_usdt)) for t in trades if str(t.side).upper() == "SHORT"))

        return {
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor if profit_factor != float("inf") else 999999.0,
            "max_drawdown": max_dd,
            "trades_n": len(pnls),
            "wins_n": len(wins),
            "losses_n": len(losses),
            "net_pnl_sum": net,
            "net_pnl_long_sum": long_net,
            "net_pnl_short_sum": short_net,
            "long_trades_n": sum(1 for t in trades if str(t.side).upper() == "LONG"),
            "short_trades_n": sum(1 for t in trades if str(t.side).upper() == "SHORT"),
            "avg_pnl_per_trade": (net / len(pnls)) if pnls else 0.0,
            "avg_hold_ms": float(avg_hold_ms),
        }


