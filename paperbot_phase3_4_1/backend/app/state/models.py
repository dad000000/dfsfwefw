from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel


@dataclass
class L2Book:
    """Very small in-memory L2 book (price -> qty)."""

    bids: Dict[float, float] = field(default_factory=dict)
    asks: Dict[float, float] = field(default_factory=dict)
    last_update_id: int = 0
    # Local receipt timestamp (ms) for staleness/age checks
    last_event_ts_ms: int = 0
    # Exchange event timestamp (ms) from Binance depth stream (if available)
    last_exchange_ts_ms: int = 0
    resyncs: int = 0
    resync_failures: int = 0

    # For correct Binance snapshot+delta stitching:
    # False immediately after snapshot, True after first successfully applied delta.
    depth_seq_started: bool = False

    def top_levels(self, n: int = 20) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        bid_lvls = sorted(self.bids.items(), key=lambda x: x[0], reverse=True)[:n]
        ask_lvls = sorted(self.asks.items(), key=lambda x: x[0])[:n]
        return bid_lvls, ask_lvls

    def best_bid_ask(self) -> Tuple[Optional[float], Optional[float]]:
        bb = max(self.bids.keys()) if self.bids else None
        ba = min(self.asks.keys()) if self.asks else None
        return bb, ba


class TradePrint(BaseModel):
    ts_ms: int
    price: float
    qty: float
    is_buyer_maker: bool  # True => sell-initiated on Binance
    side: str             # "BUY" or "SELL"


class MarkPrice(BaseModel):
    ts_ms: int
    mark_price: float


class Order(BaseModel):
    order_id: str
    ts_ms: int
    symbol: str
    side: str  # BUY/SELL
    price: float
    qty: float

    # queue + partial fill fields (for top-of-book simulation)
    filled_qty: float = 0.0
    remaining_qty: float = 0.0
    queue_ahead_qty: float = 0.0

    status: str  # NEW / PARTIALLY_FILLED / FILLED / CANCELED / REJECTED
    reject_reason: Optional[str] = None


class Fill(BaseModel):
    fill_id: str
    order_id: str
    ts_ms: int
    symbol: str
    side: str
    price: float
    qty: float
    fee_usdt: float
    liquidity: str  # MAKER/TAKER


class ClosedTrade(BaseModel):
    trade_id: str
    side: str  # LONG or SHORT
    qty: float
    entry_price: float
    exit_price: float
    entry_ts_ms: int
    exit_ts_ms: int
    realized_pnl_usdt: float
    fees_usdt: float

    @property
    def holding_ms(self) -> int:
        return max(0, int(self.exit_ts_ms) - int(self.entry_ts_ms))


class AccountState(BaseModel):
    cash_balance_usdt: float
    position_qty: float
    avg_entry_price: float
    realized_pnl_usdt: float
    unrealized_pnl_usdt: float
    fees_paid_usdt: float
    equity_usdt: float
    notional_usdt: float
    margin_used_usdt: float

