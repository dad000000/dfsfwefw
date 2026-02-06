from __future__ import annotations

from dataclasses import dataclass

from ..state.models import AccountState


@dataclass
class PnlResult:
    account: AccountState
    realized_delta: float


def apply_linear_usdm_fill(
    account: AccountState,
    side: str,
    price: float,
    qty: float,
    fee_usdt: float,
    leverage: float,
) -> PnlResult:
    """
    Linear USD-M futures position accounting (paper).
    - position_qty signed: + long, - short
    - avg_entry_price is positive for non-flat positions
    - realized_pnl_usdt accumulates only on closing lots
    - cash_balance_usdt tracks balance after realized pnl and fees

    This is a simplified model intended for paper trading & dashboard PnL.
    """
    price = float(price)
    qty = float(qty)
    fee_usdt = float(fee_usdt)

    pos = float(account.position_qty)
    avg = float(account.avg_entry_price)
    cash = float(account.cash_balance_usdt)

    # deduct fee immediately
    cash -= fee_usdt
    fees_paid = float(account.fees_paid_usdt) + fee_usdt

    fill_signed = qty if side.upper() == "BUY" else -qty

    realized = 0.0

    # If flat -> open new
    if abs(pos) < 1e-12:
        new_pos = fill_signed
        new_avg = price
        new_realized_total = float(account.realized_pnl_usdt)
        return PnlResult(
            account=account.model_copy(update={
                "cash_balance_usdt": cash,
                "fees_paid_usdt": fees_paid,
                "position_qty": new_pos,
                "avg_entry_price": new_avg,
                "realized_pnl_usdt": new_realized_total,
            }),
            realized_delta=0.0,
        )

    # Same direction add
    if (pos > 0 and fill_signed > 0) or (pos < 0 and fill_signed < 0):
        new_pos = pos + fill_signed
        # VWAP average price
        new_avg = (abs(pos) * avg + abs(fill_signed) * price) / max(abs(new_pos), 1e-12)
        new_realized_total = float(account.realized_pnl_usdt)
        return PnlResult(
            account=account.model_copy(update={
                "cash_balance_usdt": cash,
                "fees_paid_usdt": fees_paid,
                "position_qty": new_pos,
                "avg_entry_price": new_avg,
                "realized_pnl_usdt": new_realized_total,
            }),
            realized_delta=0.0,
        )

    # Opposite direction => closing (partial or full or flip)
    # closing_qty is min(abs(pos), abs(fill_signed))
    closing_qty = min(abs(pos), abs(fill_signed))

    if closing_qty > 0:
        if pos > 0:
            # closing long with sell (fill_signed negative)
            realized = closing_qty * (price - avg)
        else:
            # closing short with buy (fill_signed positive)
            realized = closing_qty * (avg - price)

        cash += realized

    new_pos = pos + fill_signed  # may flip sign
    new_realized_total = float(account.realized_pnl_usdt) + realized

    if abs(new_pos) < 1e-12:
        # back to flat
        new_avg = 0.0
    else:
        # if flipped, the remaining open portion has entry price = fill price
        # if partially closed but still same original sign, keep avg
        if (pos > 0 and new_pos > 0) or (pos < 0 and new_pos < 0):
            new_avg = avg
        else:
            new_avg = price

    return PnlResult(
        account=account.model_copy(update={
            "cash_balance_usdt": cash,
            "fees_paid_usdt": fees_paid,
            "position_qty": new_pos,
            "avg_entry_price": new_avg,
            "realized_pnl_usdt": new_realized_total,
        }),
        realized_delta=realized,
    )

