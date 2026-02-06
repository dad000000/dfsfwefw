from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from typing import List, Tuple

from ..config import Settings
from ..state.models import ClosedTrade, Order, TradePrint
from ..utils.time_ms import now_ms
from ..state.store import Store
from .fills import new_fill
from .pnl import apply_linear_usdm_fill
from .trades import TradeTracker

log = logging.getLogger("paper.matching_engine")


@dataclass
class EngineStats:
    post_only_rejects: int = 0
    fills: int = 0
    partial_fills: int = 0
    synth_events: int = 0


class PaperMatchingEngine:
    """Maker-only paper matching engine.

    Key points:
      - Post-only: we reject orders that would cross the current spread.
      - Fills are simulated using:
          (A) Synthetic top-of-book consumption inferred from depth updates (default),
          (B) aggTrade tape (optional, to avoid double counting).

    IMPORTANT:
      - This is a *paper* simulation. It approximates queueing and executions.
      - It supports *price improvement* orders inside the spread (still maker-only):
          BUY with price > best_bid and < best_ask
          SELL with price < best_ask and > best_bid
        These orders are treated as "top-of-book" with zero queue ahead.
    """

    def __init__(self, store: Store, settings: Settings) -> None:
        self.store = store
        self.s = settings
        self.stats = EngineStats()
        self._tracker = TradeTracker()

        # State for synthetic consumption inference from best bid/ask sizes
        self._prev_book_ts_ms: int = 0
        self._prev_bb: float | None = None
        self._prev_bq: float = 0.0
        self._prev_ba: float | None = None
        self._prev_aq: float = 0.0

    def _now_ms(self) -> int:
        return int(now_ms())

    def _eps(self) -> float:
        return max(float(self.store.tick_size) * 0.25, 1e-9)

    async def place_limit_post_only(self, side: str, price: float, qty: float) -> Order:
        """Place a maker-only limit in the paper engine."""
        order_id = str(uuid.uuid4())
        now = self._now_ms()

        async with self.store.lock:
            bb, ba = self.store.book.best_bid_ask()
            eps = self._eps()

            # Post-only reject if crosses
            if ba is not None and side == "BUY" and float(price) >= float(ba) - eps:
                o = Order(
                    order_id=order_id,
                    ts_ms=now,
                    symbol=self.s.SYMBOL,
                    side=side,
                    price=float(price),
                    qty=float(qty),
                    filled_qty=0.0,
                    remaining_qty=float(qty),
                    queue_ahead_qty=0.0,
                    status="REJECTED",
                    reject_reason="POST_ONLY_REJECT",
                )
                self.store.orders[o.order_id] = o
                self.store.recent_rejects.append(now)
                self.store.on_paper_reject_locked(now)
                self.store.order_rate.append(now)
                self.stats.post_only_rejects += 1
                return o

            if bb is not None and side == "SELL" and float(price) <= float(bb) + eps:
                o = Order(
                    order_id=order_id,
                    ts_ms=now,
                    symbol=self.s.SYMBOL,
                    side=side,
                    price=float(price),
                    qty=float(qty),
                    filled_qty=0.0,
                    remaining_qty=float(qty),
                    queue_ahead_qty=0.0,
                    status="REJECTED",
                    reject_reason="POST_ONLY_REJECT",
                )
                self.store.orders[o.order_id] = o
                self.store.recent_rejects.append(now)
                self.store.on_paper_reject_locked(now)
                self.store.order_rate.append(now)
                self.stats.post_only_rejects += 1
                return o

            # Queue estimate:
            # - If joining the exchange best level, assume some queue ahead.
            # - If price improves inside the spread, treat queue_ahead as 0 (we become top).
            q_factor = float(getattr(self.s, "QUEUE_AHEAD_FACTOR", 1.0))
            q_factor = max(0.0, min(q_factor, 1.0))

            queue_ahead = 0.0
            if bb is not None and ba is not None:
                bb_f = float(bb)
                ba_f = float(ba)
                p = float(price)

                inside_spread = (bb_f + eps < p < ba_f - eps)
                if not inside_spread:
                    if side == "BUY" and abs(p - bb_f) <= eps:
                        queue_ahead = float(self.store.book.bids.get(bb_f, 0.0)) * q_factor
                    if side == "SELL" and abs(p - ba_f) <= eps:
                        queue_ahead = float(self.store.book.asks.get(ba_f, 0.0)) * q_factor

            o = Order(
                order_id=order_id,
                ts_ms=now,
                symbol=self.s.SYMBOL,
                side=side,
                price=float(price),
                qty=float(qty),
                filled_qty=0.0,
                remaining_qty=float(qty),
                queue_ahead_qty=float(queue_ahead),
                status="NEW",
                reject_reason=None,
            )
            self.store.orders[o.order_id] = o
            self.store.order_rate.append(now)
            return o

    async def cancel_all(self) -> None:
        canceled = 0
        now = self._now_ms()
        async with self.store.lock:
            for oid, o in list(self.store.orders.items()):
                if o.status in ("NEW", "PARTIALLY_FILLED"):
                    self.store.orders[oid] = o.model_copy(update={"status": "CANCELED"})
                    self.store.on_paper_cancel_locked(o.side, now)
                    canceled += 1
        log.info("cancel_all canceled=%d", canceled)

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel a single open order by id (NEW/PARTIALLY_FILLED)."""
        now = self._now_ms()
        async with self.store.lock:
            o = self.store.orders.get(order_id)
            if not o:
                return False
            if o.status not in ("NEW", "PARTIALLY_FILLED"):
                return False
            self.store.orders[order_id] = o.model_copy(update={"status": "CANCELED"})
            self.store.on_paper_cancel_locked(o.side, now)
            return True

    
    async def emergency_exit(self, side: str, price: float, qty: float, *, reason: str = "EMERGENCY") -> None:
        """Simulated emergency exit INSIDE the paper model (never sent to exchange).

        This directly applies a fill at the provided price and updates PnL/trade tracking.
        Intended as a last resort to avoid getting 'stuck' when maker-only exits don't fill.
        """
        now = self._now_ms()
        side = str(side).upper()
        price = float(price)
        qty = abs(float(qty))

        if qty <= 0 or price <= 0 or side not in ("BUY", "SELL"):
            return

        fee_bps = float(getattr(self.s, "MAKER_FEE_BPS", 0.0) or 0.0)
        fee_rate = max(0.0, fee_bps / 10000.0)
        fee = abs(price * qty) * fee_rate  # conservative: still charge maker fee

        async with self.store.lock:
            before_pos = float(self.store.account.position_qty)

            # Apply PnL/accounting
            res = apply_linear_usdm_fill(
                self.store.account,
                side=side,
                price=float(price),
                qty=float(qty),
                fee_usdt=float(fee),
                leverage=float(getattr(self.s, "LEVERAGE", 1.0) or 1.0),
            )
            self.store.account = res.account

            # Record a synthetic fill
            f = new_fill(
                order_id=f"EMERGENCY_EXIT:{reason}",
                symbol=self.s.SYMBOL,
                side=side,
                price=float(price),
                qty=float(qty),
                fee_usdt=float(fee),
                liquidity="EMERGENCY",
                ts_ms=int(now),
            )
            self.store.fills.appendleft(f)
            self.store.on_paper_fill_locked(side, int(now))
            self.store.on_paper_fill_details_locked(f)

            after_pos = float(self.store.account.position_qty)

            closed: List[ClosedTrade] = []
            self._tracker.on_fill(
                before_pos=float(before_pos),
                after_pos=float(after_pos),
                fill_price=float(price),
                fill_qty=float(qty),
                realized_delta=float(res.realized_delta),
                fee_total=float(fee),
                ts_ms=int(now),
                closed_trade_sink=closed,
            )
            for ct in closed:
                self.store.closed_trades.appendleft(ct)
                self.store.on_paper_closed_trade_locked(ct)

            self.store._append_series_locked()

    async def on_book_tick(self) -> None:
        """Called after each depth update.

        Responsibilities:
          1) Maker-only integrity: if market moves through our price, fill remaining qty.
          2) Clamp queue_ahead for orders sitting exactly at exchange best bid/ask.
          3) Synthetic consumption: infer aggressor volume from best size reductions
             and fill eligible maker orders (including inside-spread improved orders).
        """
        async with self.store.lock:
            bb, ba = self.store.book.best_bid_ask()
            if bb is None or ba is None:
                return

            eps = self._eps()
            bb = float(bb)
            ba = float(ba)
            bb_qty = float(self.store.book.bids.get(bb, 0.0))
            ba_qty = float(self.store.book.asks.get(ba, 0.0))
            book_ts = int(self.store.book.last_event_ts_ms or self._now_ms())

            # 1) If market moved through our resting price, simulate a maker fill (better than cancel).
            for oid, o in list(self.store.orders.items()):
                if o.status not in ("NEW", "PARTIALLY_FILLED"):
                    continue
                if float(o.remaining_qty) <= 0:
                    continue

                if o.side == "BUY" and float(o.price) >= ba - eps:
                    await self._apply_fill_locked(
                        order=o,
                        order_id=oid,
                        fill_price=float(o.price),
                        fill_qty=float(o.remaining_qty),
                        ts_ms=book_ts,
                    )
                    continue

                if o.side == "SELL" and float(o.price) <= bb + eps:
                    await self._apply_fill_locked(
                        order=o,
                        order_id=oid,
                        fill_price=float(o.price),
                        fill_qty=float(o.remaining_qty),
                        ts_ms=book_ts,
                    )
                    continue

            # 2) Clamp queue ahead at exchange best prices (cancellations ahead can move us forward).
            for oid, o in list(self.store.orders.items()):
                if o.status not in ("NEW", "PARTIALLY_FILLED"):
                    continue

                if o.side == "BUY" and abs(float(o.price) - bb) <= eps:
                    new_ahead = min(float(o.queue_ahead_qty), bb_qty)
                    if new_ahead != float(o.queue_ahead_qty):
                        self.store.orders[oid] = o.model_copy(update={"queue_ahead_qty": new_ahead})

                if o.side == "SELL" and abs(float(o.price) - ba) <= eps:
                    new_ahead = min(float(o.queue_ahead_qty), ba_qty)
                    if new_ahead != float(o.queue_ahead_qty):
                        self.store.orders[oid] = o.model_copy(update={"queue_ahead_qty": new_ahead})

            # 3) Synthetic consumption from book
            synth_enabled = bool(getattr(self.s, "SYNTH_FILL_FROM_BOOK", True))
            if synth_enabled and book_ts != self._prev_book_ts_ms:
                # Bids: if size decreases at same bb => consumed
                if self._prev_bb is not None:
                    prev_bb = float(self._prev_bb)
                    prev_bq = float(self._prev_bq)

                    consumed = 0.0
                    top_price = bb

                    if abs(bb - prev_bb) <= eps:
                        consumed = max(0.0, prev_bq - bb_qty)
                        top_price = bb
                    elif bb < prev_bb - eps:
                        # best bid moved DOWN => previous level likely removed (often by consumption)
                        consumed = max(0.0, prev_bq)
                        top_price = prev_bb

                    if consumed > 0:
                        await self._consume_top_locked(
                            want_side="BUY",
                            top_price=float(top_price),
                            consume_qty=float(consumed),
                            ts_ms=book_ts,
                        )
                        self.stats.synth_events += 1

                # Asks
                if self._prev_ba is not None:
                    prev_ba = float(self._prev_ba)
                    prev_aq = float(self._prev_aq)

                    consumed = 0.0
                    top_price = ba

                    if abs(ba - prev_ba) <= eps:
                        consumed = max(0.0, prev_aq - ba_qty)
                        top_price = ba
                    elif ba > prev_ba + eps:
                        # best ask moved UP => previous level likely removed (often by consumption)
                        consumed = max(0.0, prev_aq)
                        top_price = prev_ba

                    if consumed > 0:
                        await self._consume_top_locked(
                            want_side="SELL",
                            top_price=float(top_price),
                            consume_qty=float(consumed),
                            ts_ms=book_ts,
                        )
                        self.stats.synth_events += 1

            # Update prev snapshot
            self._prev_book_ts_ms = book_ts
            self._prev_bb = bb
            self._prev_bq = bb_qty
            self._prev_ba = ba
            self._prev_aq = ba_qty

    async def on_market_trade(self, t: TradePrint) -> None:
        """Fill from aggTrade tape (optional).

        If SYNTH_FILL_FROM_BOOK is enabled (default), we do *not* use the tape
        to avoid double-counting. If you disable synth fills, the tape becomes
        the driver for queue consumption.
        """
        if bool(getattr(self.s, "SYNTH_FILL_FROM_BOOK", True)):
            return

        async with self.store.lock:
            bb, ba = self.store.book.best_bid_ask()
            if bb is None or ba is None:
                return

            eps = self._eps()
            bb = float(bb)
            ba = float(ba)

            if t.side == "SELL":
                # aggressive sell consumes bids -> fills BUY makers
                if float(t.price) > bb + eps:
                    return
                want_side = "BUY"
                top_price = bb
                top_qty_now = float(self.store.book.bids.get(bb, 0.0))
            else:
                # aggressive buy consumes asks -> fills SELL makers
                if float(t.price) < ba - eps:
                    return
                want_side = "SELL"
                top_price = ba
                top_qty_now = float(self.store.book.asks.get(ba, 0.0))

            tape_qty = float(t.qty)
            if tape_qty <= 0 or top_qty_now <= 0:
                return

            await self._consume_top_locked(
                want_side=want_side,
                top_price=float(top_price),
                consume_qty=min(tape_qty, top_qty_now),
                ts_ms=int(t.ts_ms),
            )

    async def _consume_top_locked(self, want_side: str, top_price: float, consume_qty: float, ts_ms: int) -> None:
        """Consume inferred aggressor volume and fill eligible maker orders.

        want_side:
          - "BUY"  => fill BUY maker orders (sell pressure consuming bids)
          - "SELL" => fill SELL maker orders (buy pressure consuming asks)

        Supports price improvement orders inside the spread:
          - BUY orders priced ABOVE exchange best bid (but still < best ask) get priority.
          - SELL orders priced BELOW exchange best ask (but still > best bid) get priority.
        """
        remaining = float(consume_qty)
        if remaining <= 0:
            return

        eps = self._eps()
        bb, ba = self.store.book.best_bid_ask()
        bb = float(bb) if bb is not None else None
        ba = float(ba) if ba is not None else None

        eligible: List[Tuple[str, Order]] = []
        for oid, o in self.store.orders.items():
            if o.status not in ("NEW", "PARTIALLY_FILLED"):
                continue
            if float(o.remaining_qty) <= 0:
                continue
            if o.side != want_side:
                continue

            p = float(o.price)

            if want_side == "BUY":
                # maker-only: can't be crossing ask
                if ba is not None and p >= ba - eps:
                    continue
                # must be at/above the consumed top bid price (or better)
                if p + eps < float(top_price):
                    continue
            else:
                # SELL
                if bb is not None and p <= bb + eps:
                    continue
                # must be at/below the consumed top ask price (or better)
                if p - eps > float(top_price):
                    continue

            eligible.append((oid, o))

        if not eligible:
            return

        # Priority:
        #  - BUY: higher price first, then FIFO
        #  - SELL: lower price first, then FIFO
        if want_side == "BUY":
            eligible.sort(key=lambda x: (-float(x[1].price), int(x[1].ts_ms)))
        else:
            eligible.sort(key=lambda x: (float(x[1].price), int(x[1].ts_ms)))

        for oid, o in eligible:
            if remaining <= 0:
                break

            # Orders exactly at top_price have queue ahead; improved orders treated as top (queue 0)
            if abs(float(o.price) - float(top_price)) <= eps:
                o_ahead = min(float(o.queue_ahead_qty), remaining)
            else:
                o_ahead = 0.0

            if o_ahead > 0:
                take = min(o_ahead, remaining)
                o_ahead -= take
                remaining -= take

            if remaining <= 0:
                if o_ahead != float(o.queue_ahead_qty):
                    self.store.orders[oid] = o.model_copy(update={"queue_ahead_qty": o_ahead})
                continue

            fill_qty = min(float(o.remaining_qty), remaining)
            if fill_qty <= 0:
                if o_ahead != float(o.queue_ahead_qty):
                    self.store.orders[oid] = o.model_copy(update={"queue_ahead_qty": o_ahead})
                continue

            await self._apply_fill_locked(
                order=o,
                order_id=oid,
                fill_price=float(o.price),
                fill_qty=float(fill_qty),
                ts_ms=int(ts_ms),
            )
            remaining -= fill_qty

            # Update queue ahead (only meaningful at top_price)
            oo = self.store.orders.get(oid)
            if (
                oo
                and oo.status in ("NEW", "PARTIALLY_FILLED")
                and abs(float(oo.price) - float(top_price)) <= eps
                and float(oo.queue_ahead_qty) != float(o_ahead)
            ):
                self.store.orders[oid] = oo.model_copy(update={"queue_ahead_qty": o_ahead})

    async def _apply_fill_locked(self, order: Order, order_id: str, fill_price: float, fill_qty: float, ts_ms: int) -> None:
        """Apply a maker fill against the paper account (store.lock MUST be held)."""
        fee = (self.s.MAKER_FEE_BPS / 10000.0) * (abs(float(fill_qty)) * float(fill_price))

        before_pos = float(self.store.account.position_qty)

        res = apply_linear_usdm_fill(
            account=self.store.account,
            side=order.side,
            price=float(fill_price),
            qty=float(fill_qty),
            fee_usdt=float(fee),
            leverage=self.s.LEVERAGE,
        )
        self.store.account = res.account
        after_pos = float(self.store.account.position_qty)

        await self.store._recompute_account_locked()

        new_filled = float(order.filled_qty) + float(fill_qty)
        new_remaining = max(0.0, float(order.qty) - new_filled)
        new_status = "FILLED" if new_remaining <= 1e-12 else "PARTIALLY_FILLED"

        self.store.orders[order_id] = order.model_copy(
            update={
                "filled_qty": new_filled,
                "remaining_qty": new_remaining,
                "status": new_status,
            }
        )

        f = new_fill(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            price=float(fill_price),
            qty=float(fill_qty),
            fee_usdt=float(fee),
            liquidity="MAKER",
            ts_ms=int(ts_ms),
        )
        self.store.fills.appendleft(f)

        # execution diagnostics
        # Use local time for diagnostics/probes to avoid clock drift vs exchange timestamps.
        self.store.on_paper_fill_locked(order.side, int(ts_ms))
        self.store.on_paper_fill_details_locked(f)

        self.stats.fills += 1
        if new_status == "PARTIALLY_FILLED":
            self.stats.partial_fills += 1

        closed: List[ClosedTrade] = []
        self._tracker.on_fill(
            before_pos=before_pos,
            after_pos=after_pos,
            fill_price=float(fill_price),
            fill_qty=float(fill_qty),
            realized_delta=float(res.realized_delta),
            fee_total=float(fee),
            ts_ms=int(ts_ms),
            closed_trade_sink=closed,
        )
        for ct in closed:
            self.store.closed_trades.appendleft(ct)
            self.store.on_paper_closed_trade_locked(ct)

        self.store._append_series_locked()

