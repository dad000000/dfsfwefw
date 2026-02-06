from __future__ import annotations

import uuid

from ..state.models import Fill
from ..utils.time_ms import now_ms


def _now_ms() -> int:
    return int(now_ms())


def new_fill(
    order_id: str,
    symbol: str,
    side: str,
    price: float,
    qty: float,
    fee_usdt: float,
    liquidity: str,
    ts_ms: int | None = None,
) -> Fill:
    return Fill(
        fill_id=str(uuid.uuid4()),
        order_id=order_id,
        ts_ms=int(ts_ms) if ts_ms is not None else _now_ms(),
        symbol=symbol,
        side=side,
        price=float(price),
        qty=float(qty),
        fee_usdt=float(fee_usdt),
        liquidity=liquidity,
    )

