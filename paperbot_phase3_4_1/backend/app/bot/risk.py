from __future__ import annotations

import time
from dataclasses import dataclass

from ..config import Settings
from ..state.store import Store
from ..utils.time_ms import now_ms


@dataclass
class RiskStatus:
    ok: bool
    reason: str = ""


class RiskManager:
    def __init__(self, store: Store, settings: Settings) -> None:
        self.store = store
        self.s = settings
        self._daily_start_ms = int(now_ms())
        self._daily_start_equity = settings.PAPER_START_BALANCE_USDT

    async def reset_daily(self) -> None:
        async with self.store.lock:
            self._daily_start_ms = int(now_ms())
            self._daily_start_equity = float(self.store.account.equity_usdt)

    async def check(self) -> RiskStatus:
        snap = await self.store.snapshot()
        now_ms = int(snap["ts_ms"])

        book_age = snap["connectivity"]["book_last_update_age_ms"]
        if book_age is not None and int(book_age) > self.s.MAX_BOOK_STALE_MS:
            return RiskStatus(False, f"STALE_BOOK age_ms={book_age}")

        ws_age = snap["connectivity"]["ws_last_event_age_ms"]
        if ws_age is not None and int(ws_age) > self.s.MAX_EVENT_LAG_MS:
            return RiskStatus(False, f"STALE_WS age_ms={ws_age}")

        # Position sanity checks
        acct = snap.get("account", {}) or {}
        mkt = snap.get("market", {}) or {}
        pos = float(acct.get("position_qty") or 0.0)

        # 1) Notional-based hard cap (aligned with strategy cap, with small headroom).
        equity = float(acct.get("equity_usdt") or 0.0)
        cap_pct = float(getattr(self.s, "STRATEGY_MAX_NOTIONAL_PCT", 0.10) or 0.10)
        headroom = float(getattr(self.s, "RISK_MAX_POS_NOTIONAL_MULT", 1.25) or 1.25)
        max_notional = max(0.0, equity * cap_pct * headroom)

        avg = float(acct.get("avg_entry_price") or 0.0)
        mid = float(mkt.get("mid") or 0.0)
        mark = float(mkt.get("mark_price") or 0.0)
        ref_px = avg if avg > 0 else (mid if mid > 0 else mark)
        pos_notional = abs(pos) * float(ref_px) if ref_px > 0 else 0.0

        if max_notional > 0 and pos_notional > (max_notional + 1e-6):
            return RiskStatus(
                False,
                (
                    "MAX_POSITION_NOTIONAL_EXCEEDED "
                    f"pos_notional={pos_notional:.2f} max_notional={max_notional:.2f} "
                    f"pos_qty={pos:.6f} ref_px={ref_px:.2f}"
                ),
            )

        # 2) Absolute qty sanity cap (keep very high; mainly protects from symbol misconfig).
        if abs(pos) > float(self.s.MAX_POSITION_QTY) + 1e-12:
            return RiskStatus(False, f"MAX_POSITION_QTY_EXCEEDED pos={pos:.6f} max_qty={float(self.s.MAX_POSITION_QTY):.6f}")

        # open orders include partially filled
        open_orders = [o for o in snap.get("orders", []) if o.get("status") in ("NEW", "PARTIALLY_FILLED")]
        if len(open_orders) > int(self.s.MAX_OPEN_ORDERS):
            return RiskStatus(False, f"MAX_OPEN_ORDERS open={len(open_orders)}")

        cutoff = now_ms - 60_000
        async with self.store.lock:
            recent_orders = [t for t in self.store.order_rate if t >= cutoff]
            recent_rejects = [t for t in self.store.recent_rejects if t >= cutoff]

        if len(recent_orders) > int(self.s.MAX_ORDERS_PER_MIN):
            return RiskStatus(False, f"ORDER_RATE_LIMIT n={len(recent_orders)}")
        if len(recent_rejects) >= 15:
            return RiskStatus(False, f"EXCESSIVE_REJECTS n={len(recent_rejects)}")

        equity = float(snap["account"]["equity_usdt"])
        daily_loss = float(self._daily_start_equity) - equity
        if daily_loss > float(self.s.MAX_DAILY_LOSS_USDT):
            return RiskStatus(False, f"MAX_DAILY_LOSS loss={daily_loss:.2f}")

        return RiskStatus(True, "")

