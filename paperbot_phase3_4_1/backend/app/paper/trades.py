from __future__ import annotations

import uuid
from dataclasses import dataclass

from ..state.models import ClosedTrade
from ..utils.time_ms import now_ms


def _now_ms() -> int:
    return int(now_ms())


@dataclass
class OpenTrade:
    trade_id: str
    side: str  # LONG/SHORT
    entry_ts_ms: int

    # totals for reporting (VWAP)
    entry_qty_total: float = 0.0
    entry_notional_total: float = 0.0

    exit_qty_total: float = 0.0
    exit_notional_total: float = 0.0

    realized_pnl_total: float = 0.0
    fees_usdt_total: float = 0.0

    @property
    def entry_vwap(self) -> float:
        return (self.entry_notional_total / self.entry_qty_total) if self.entry_qty_total > 0 else 0.0

    @property
    def exit_vwap(self) -> float:
        return (self.exit_notional_total / self.exit_qty_total) if self.exit_qty_total > 0 else 0.0


class TradeTracker:
    """
    One "trade" = flat -> non-flat -> flat (or flip closes old trade, opens new trade).
    Accumulates:
      - realized_pnl across multiple fills/partial closes
      - entry/exit VWAP across multiple entry/exit fills
      - fees allocated proportionally on flip fills
    """

    def __init__(self) -> None:
        self.current: OpenTrade | None = None

    def _start_new(self, side: str, ts_ms: int) -> OpenTrade:
        return OpenTrade(trade_id=str(uuid.uuid4()), side=side, entry_ts_ms=ts_ms)

    def on_fill(
        self,
        before_pos: float,
        after_pos: float,
        fill_price: float,
        fill_qty: float,
        realized_delta: float,
        fee_total: float,
        ts_ms: int,
        closed_trade_sink: list[ClosedTrade],
    ) -> None:
        """
        before_pos/after_pos are signed position quantities (+: long, -: short).
        realized_delta corresponds ONLY to the closing portion (from PnL engine).
        """

        if fill_qty <= 0:
            return

        abs_before = abs(before_pos)
        abs_after = abs(after_pos)

        def sign(x: float) -> int:
            if x > 0:
                return 1
            if x < 0:
                return -1
            return 0

        s_before = sign(before_pos)
        s_after = sign(after_pos)
        flipped = (s_before != 0 and s_after != 0 and s_before != s_after)

        qty_close = 0.0
        qty_open = 0.0

        if abs_before == 0 and abs_after > 0:
            qty_open = abs_after  # from flat
        elif flipped:
            qty_close = abs_before
            qty_open = abs_after
        else:
            if abs_after > abs_before:
                qty_open = abs_after - abs_before
            elif abs_after < abs_before:
                qty_close = abs_before - abs_after

        # Fee allocation (important on flip fills)
        fee_close = 0.0
        if fill_qty > 0 and qty_close > 0:
            fee_close = fee_total * (qty_close / fill_qty)
        fee_open = fee_total - fee_close

        # Ensure we have a current trade if we're in a position
        if self.current is None:
            if qty_open > 0:
                side = "LONG" if after_pos > 0 else "SHORT"
                self.current = self._start_new(side=side, ts_ms=ts_ms)
                self.current.entry_qty_total += qty_open
                self.current.entry_notional_total += qty_open * fill_price
                self.current.fees_usdt_total += fee_open
            return

        # We have a current trade
        t = self.current

        # Closing portion
        if qty_close > 0:
            t.exit_qty_total += qty_close
            t.exit_notional_total += qty_close * fill_price
            t.realized_pnl_total += realized_delta
            t.fees_usdt_total += fee_close

        # Opening/add portion (same trade if not flipped yet)
        if qty_open > 0 and not flipped:
            t.entry_qty_total += qty_open
            t.entry_notional_total += qty_open * fill_price
            t.fees_usdt_total += fee_open

        # Close conditions:
        # - back to flat
        # - flipped (close old trade now, then open new one)
        if after_pos == 0 or flipped:
            exit_ts = max(int(ts_ms), int(t.entry_ts_ms))
            closed = ClosedTrade(
                trade_id=t.trade_id,
                side=t.side,
                qty=t.entry_qty_total if t.entry_qty_total > 0 else (t.exit_qty_total if t.exit_qty_total > 0 else abs_before),
                entry_price=t.entry_vwap if t.entry_vwap > 0 else fill_price,
                exit_price=t.exit_vwap if t.exit_vwap > 0 else fill_price,
                entry_ts_ms=t.entry_ts_ms,
                exit_ts_ms=exit_ts,
                realized_pnl_usdt=t.realized_pnl_total,
                fees_usdt=t.fees_usdt_total,
            )
            closed_trade_sink.append(closed)
            self.current = None

            # If flipped, immediately start new trade with the opening remainder
            if flipped and qty_open > 0:
                side = "LONG" if after_pos > 0 else "SHORT"
                nt = self._start_new(side=side, ts_ms=ts_ms)
                nt.entry_qty_total += qty_open
                nt.entry_notional_total += qty_open * fill_price
                nt.fees_usdt_total += fee_open
                self.current = nt

