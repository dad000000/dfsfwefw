from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ..bot.strategy import MicroScalpStrategy
from ..config import Settings
from ..paper.matching_engine import PaperMatchingEngine
from ..state.models import ClosedTrade, Order, TradePrint
from ..state.store import Store

MS_DAY = 24 * 60 * 60 * 1000


@dataclass
class EquityPoint:
    ts_ms: int
    equity_usdt: float
    cash_balance_usdt: float
    position_qty: float
    avg_entry_price: float
    realized_pnl_usdt: float
    unrealized_pnl_usdt: float
    fees_paid_usdt: float


def _iter_ndjson(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                yield json.loads(s)
            except json.JSONDecodeError:
                # Skip corrupted lines (rare if process is killed mid-write)
                continue


def _detect_symbol(path: str, default: str = "ETHUSDT") -> str:
    for ev in _iter_ndjson(path):
        if ev.get("type") == "frame":
            sym = ev.get("symbol")
            if isinstance(sym, str) and sym:
                return sym.upper()
    return default


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _f(x: Any) -> float:
    try:
        v = float(x)
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return 0.0


def _i(x: Any) -> int:
    try:
        return int(x)
    except Exception:
        return 0


def _calc_max_drawdown(equity: List[EquityPoint]) -> Tuple[float, float]:
    peak: Optional[float] = None
    max_dd = 0.0
    max_dd_pct = 0.0
    for p in equity:
        eq = float(p.equity_usdt)
        if peak is None or eq > peak:
            peak = eq
        if peak and peak > 0:
            dd = peak - eq
            if dd > max_dd:
                max_dd = dd
                max_dd_pct = dd / peak
    return max_dd, max_dd_pct


def _profit_factor_net(trades: List[ClosedTrade]) -> float:
    prof = 0.0
    loss = 0.0
    for t in trades:
        net = float(t.realized_pnl_usdt) - float(t.fees_usdt)
        if net >= 0:
            prof += net
        else:
            loss += -net
    if loss <= 0:
        return float("inf") if prof > 0 else 0.0
    return prof / loss


def _write_trades_csv(path: str, trades: List[ClosedTrade]) -> None:
    fields = [
        "trade_id",
        "side",
        "qty",
        "entry_price",
        "exit_price",
        "entry_ts_ms",
        "exit_ts_ms",
        "hold_ms",
        "realized_pnl_usdt",
        "fees_usdt",
        "net_after_fees_usdt",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for t in trades:
            net = float(t.realized_pnl_usdt) - float(t.fees_usdt)
            hold_ms = max(0, int(t.exit_ts_ms) - int(t.entry_ts_ms))
            w.writerow(
                {
                    "trade_id": t.trade_id,
                    "side": t.side,
                    "qty": float(t.qty),
                    "entry_price": float(t.entry_price),
                    "exit_price": float(t.exit_price),
                    "entry_ts_ms": int(t.entry_ts_ms),
                    "exit_ts_ms": int(t.exit_ts_ms),
                    "hold_ms": hold_ms,
                    "realized_pnl_usdt": float(t.realized_pnl_usdt),
                    "fees_usdt": float(t.fees_usdt),
                    "net_after_fees_usdt": net,
                }
            )


def _write_equity_csv(path: str, equity: List[EquityPoint]) -> None:
    fields = [
        "ts_ms",
        "equity_usdt",
        "cash_balance_usdt",
        "position_qty",
        "avg_entry_price",
        "realized_pnl_usdt",
        "unrealized_pnl_usdt",
        "fees_paid_usdt",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for p in equity:
            w.writerow(
                {
                    "ts_ms": int(p.ts_ms),
                    "equity_usdt": float(p.equity_usdt),
                    "cash_balance_usdt": float(p.cash_balance_usdt),
                    "position_qty": float(p.position_qty),
                    "avg_entry_price": float(p.avg_entry_price),
                    "realized_pnl_usdt": float(p.realized_pnl_usdt),
                    "unrealized_pnl_usdt": float(p.unrealized_pnl_usdt),
                    "fees_paid_usdt": float(p.fees_paid_usdt),
                }
            )


def _write_orders_csv(path: str, orders: List[Tuple[str, Order]]) -> None:
    """Export orders to CSV.

    NOTE: The Order model in this project is intentionally small. For offline
    analysis we export only fields that exist and are meaningful for maker fills.
    """
    fields = [
        "order_id",
        "symbol",
        "side",
        "price",
        "qty",
        "filled_qty",
        "remaining_qty",
        "queue_ahead_qty",
        "status",
        "ts_ms",
        "reject_reason",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for oid, o in orders:
            w.writerow(
                {
                    "order_id": oid,
                    "symbol": o.symbol,
                    "side": o.side,
                    "price": float(o.price),
                    "qty": float(o.qty),
                    "filled_qty": float(o.filled_qty),
                    "remaining_qty": float(o.remaining_qty),
                    "queue_ahead_qty": float(o.queue_ahead_qty),
                    "status": o.status,
                    "ts_ms": int(o.ts_ms),
                    "reject_reason": o.reject_reason or "",
                }
            )


async def run_backtest(
    log_path: str,
    out_dir: str,
    *,
    start_balance_usdt: Optional[float] = None,
    synth_fill_from_book: Optional[bool] = None,
    queue_ahead_factor: Optional[float] = None,
    max_frames: Optional[int] = None,
    equity_every_ms: int = 1000,
    progress_every_s: float = 3.0,
    t0_ms: Optional[int] = None,
    t1_ms: Optional[int] = None,
    settings_overrides: Optional[Dict[str, Any]] = None,
    write_outputs: bool = True,
) -> Dict[str, Any]:
    symbol = _detect_symbol(log_path)

    kwargs: Dict[str, Any] = {"SYMBOL": symbol, "RECORDER_ENABLED": False}
    if settings_overrides:
        for k, v in settings_overrides.items():
            if k is None:
                continue
            kwargs[str(k)] = v
    if start_balance_usdt is not None:
        kwargs["PAPER_START_BALANCE_USDT"] = float(start_balance_usdt)
    if synth_fill_from_book is not None:
        kwargs["SYNTH_FILL_FROM_BOOK"] = bool(synth_fill_from_book)
    if queue_ahead_factor is not None:
        try:
            qf = float(queue_ahead_factor)
            if math.isfinite(qf):
                kwargs["QUEUE_AHEAD_FACTOR"] = max(0.0, min(qf, 1.0))
        except Exception:
            pass
    s = Settings(**kwargs)

    store = Store(s)
    engine = PaperMatchingEngine(store, s)
    strat = MicroScalpStrategy(store, engine, s)

    # Offline: pretend we're connected and running.
    await store.set_bot_running(True)
    await store.update_ws_status(True, 0, 0)

    # Replay clock: make all time-based logic deterministic.
    clock: Dict[str, int] = {"ts_ms": 0}

    def now_ms() -> int:
        return int(clock["ts_ms"])

    store._now_ms = now_ms  # type: ignore[attr-defined]

    if write_outputs:
        _ensure_dir(out_dir)

    equity: List[EquityPoint] = []
    last_eq_ts = -10**18
    frames = 0
    first_ts: Optional[int] = None
    last_ts: Optional[int] = None

    loop = asyncio.get_running_loop()
    last_progress = loop.time()

    for ev in _iter_ndjson(log_path):
        if ev.get("type") != "frame":
            continue

        ts_ms = _i(ev.get("ts_ms"))
        if ts_ms <= 0:
            continue
        if t0_ms is not None and ts_ms < int(t0_ms):
            continue
        if t1_ms is not None and ts_ms >= int(t1_ms):
            break

        clock["ts_ms"] = ts_ms
        if first_ts is None:
            first_ts = ts_ms
        last_ts = ts_ms

        # schema compatibility: prefer Phase 2.1 "book", accept "top" as alias
        book = ev.get("book") or ev.get("top") or {}
        bid = _f(book.get("bid"))
        ask = _f(book.get("ask"))
        bid_qty = _f(book.get("bid_qty"))
        ask_qty = _f(book.get("ask_qty"))
        tick_size_raw = book.get("tick_size")
        tick_size = _f(tick_size_raw) if tick_size_raw is not None else None
        await store.set_top_of_book(ts_ms, bid, bid_qty, ask, ask_qty, tick_size=tick_size)
        # keep ws_last_event_ts_ms aligned to replay clock so ws_age_ms stays sane
        await store.update_ws_status(True, 0, ts_ms)

        # schema compatibility: prefer Phase 2.1 flat keys, accept nested "mark" dict
        mark_price = _f(ev.get("mark_price"))
        mark_ts = _i(ev.get("mark_ts_ms")) or ts_ms
        if mark_price <= 0 and isinstance(ev.get("mark"), dict):
            mk = ev.get("mark") or {}
            mark_price = _f(mk.get("mark_price"))
            mark_ts = _i(mk.get("ts_ms")) or mark_ts
        # Fallback: if mark is missing in the log, use mid as a stable proxy
        if mark_price <= 0:
            mid = _f(book.get("mid"))
            if mid <= 0 and bid > 0 and ask > 0:
                mid = (bid + ask) / 2.0
            if mid > 0:
                mark_price = mid
                mark_ts = ts_ms

        if mark_price > 0:
            await store.on_mark(mark_ts, mark_price)

        lt = ev.get("last_trade")
        if isinstance(lt, dict):
            t_ts = _i(lt.get("ts_ms"))
            t_price = _f(lt.get("price"))
            t_qty = _f(lt.get("qty"))
            t_side = str(lt.get("side") or "").upper()
            if t_ts > 0 and t_price > 0 and t_qty > 0 and t_side in ("BUY", "SELL"):
                # Convention used throughout the codebase:
                #   side == "SELL" implies aggressive sell (buyer is maker)
                tp = TradePrint(
                    ts_ms=t_ts,
                    price=t_price,
                    qty=t_qty,
                    side=t_side,
                    is_buyer_maker=(t_side == "SELL"),
                )
                await store.on_trade_print(tp)
                # If SYNTH_FILL_FROM_BOOK is disabled, the tape drives fills.
                await engine.on_market_trade(tp)

        # Consume book changes -> maker fills for outstanding orders
        await engine.on_book_tick()

        # Let the strategy react & place/cancel orders
        await strat._step()

        frames += 1
        if max_frames is not None and frames >= max_frames:
            break

        if equity_every_ms > 0 and ts_ms - last_eq_ts >= equity_every_ms:
            async with store.lock:
                a = store.account
                equity.append(
                    EquityPoint(
                        ts_ms=ts_ms,
                        equity_usdt=float(a.equity_usdt),
                        cash_balance_usdt=float(a.cash_balance_usdt),
                        position_qty=float(a.position_qty),
                        avg_entry_price=float(a.avg_entry_price),
                        realized_pnl_usdt=float(a.realized_pnl_usdt),
                        unrealized_pnl_usdt=float(a.unrealized_pnl_usdt),
                        fees_paid_usdt=float(a.fees_paid_usdt),
                    )
                )
            last_eq_ts = ts_ms

        now = loop.time()
        if progress_every_s > 0 and now - last_progress >= progress_every_s:
            async with store.lock:
                a = store.account
                open_orders = len([o for o in store.orders.values() if o.status in ("NEW", "PARTIALLY_FILLED")])
                print(
                    f"[backtest] frames={frames} ts_ms={ts_ms} eq={a.equity_usdt:.2f} pos={a.position_qty:.4f} open_orders={open_orders} closed_trades={len(store.closed_trades)} cancels={store.exec_cancels_total} rejects={store.exec_rejects_total}",
                    flush=True,
                )
            last_progress = now

    # Final equity point
    if last_ts is not None:
        async with store.lock:
            a = store.account
            equity.append(
                EquityPoint(
                    ts_ms=last_ts,
                    equity_usdt=float(a.equity_usdt),
                    cash_balance_usdt=float(a.cash_balance_usdt),
                    position_qty=float(a.position_qty),
                    avg_entry_price=float(a.avg_entry_price),
                    realized_pnl_usdt=float(a.realized_pnl_usdt),
                    unrealized_pnl_usdt=float(a.unrealized_pnl_usdt),
                    fees_paid_usdt=float(a.fees_paid_usdt),
                )
            )

    async with store.lock:
        trades = list(store.closed_trades)
        orders_items = list(store.orders.items())
        cancels = int(store.exec_cancels_total)
        rejects = int(store.exec_rejects_total)

    trades_sorted = sorted(trades, key=lambda t: int(t.entry_ts_ms))

    hold_ms_vals: List[int] = [max(0, int(t.exit_ts_ms) - int(t.entry_ts_ms)) for t in trades_sorted]
    net_per_trade: List[float] = [float(t.realized_pnl_usdt) - float(t.fees_usdt) for t in trades_sorted]
    win_rate = (
        (sum(1 for n in net_per_trade if n > 0) / len(net_per_trade))
        if net_per_trade
        else 0.0
    )
    avg_hold_ms = (sum(hold_ms_vals) / len(hold_ms_vals)) if hold_ms_vals else 0.0
    median_hold_ms = 0.0
    if hold_ms_vals:
        hs = sorted(hold_ms_vals)
        mid = len(hs) // 2
        median_hold_ms = float(hs[mid]) if len(hs) % 2 == 1 else float(hs[mid - 1] + hs[mid]) / 2.0

    gross = sum(float(t.realized_pnl_usdt) for t in trades_sorted)
    fees = sum(float(t.fees_usdt) for t in trades_sorted)
    net = sum((float(t.realized_pnl_usdt) - float(t.fees_usdt)) for t in trades_sorted)

    pf = _profit_factor_net(trades_sorted)
    max_dd, max_dd_pct = _calc_max_drawdown(equity)

    duration_ms = 0
    if first_ts is not None and last_ts is not None and last_ts >= first_ts:
        duration_ms = last_ts - first_ts
    days = (duration_ms / MS_DAY) if duration_ms > 0 else 0.0
    trades_per_day = (len(trades_sorted) / days) if days > 0 else float(len(trades_sorted))

    minutes = duration_ms / 60000.0 if duration_ms > 0 else 0.0
    cancels_per_min = (cancels / minutes) if minutes > 0 else float(cancels)
    rejects_per_min = (rejects / minutes) if minutes > 0 else float(rejects)

    equity_start = float(equity[0].equity_usdt) if equity else float(store.account.equity_usdt)
    equity_end = float(equity[-1].equity_usdt) if equity else float(store.account.equity_usdt)

    summary: Dict[str, Any] = {
        "symbol": symbol,
        "frames": frames,
        "t_start_ms": first_ts,
        "t_end_ms": last_ts,
        "duration_ms": duration_ms,
        "trades": len(trades_sorted),
        "win_rate_net": win_rate,
        "avg_hold_ms": avg_hold_ms,
        "median_hold_ms": median_hold_ms,
        "max_hold_ms": (max(hold_ms_vals) if hold_ms_vals else 0),
        "equity_start_usdt": equity_start,
        "equity_end_usdt": equity_end,
        "gross_realized_pnl_usdt": gross,
        "fees_usdt": fees,
        "net_after_fees_usdt": net,
        "profit_factor_net": pf,
        "max_dd_usdt": max_dd,
        "max_dd_pct": max_dd_pct,
        "trades_per_day": trades_per_day,
        "cancels_total": cancels,
        "cancels_per_min": cancels_per_min,
        "rejects_total": rejects,
        "rejects_per_min": rejects_per_min,
    }

    if write_outputs:
        _write_trades_csv(os.path.join(out_dir, "trades.csv"), trades_sorted)
        _write_orders_csv(os.path.join(out_dir, "orders.csv"), orders_items)
        _write_equity_csv(os.path.join(out_dir, "equity.csv"), equity)

        with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)

    return summary


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 2.2 offline backtest runner (maker-fill approx)")
    p.add_argument("--log", required=True, help="Path to NDJSON recording (frames)")
    p.add_argument("--out", required=True, help="Output directory")
    p.add_argument("--start-balance", type=float, default=None, help="Override PAPER_START_BALANCE_USDT")
    p.add_argument(
        "--synth-fill-from-book",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="If set, overrides SYNTH_FILL_FROM_BOOK (True=depth-based fills; False=tape-based)",
    )
    p.add_argument(
        "--queue-ahead-factor",
        type=float,
        default=None,
        help="Override QUEUE_AHEAD_FACTOR in [0..1] (queue proxy when joining best bid/ask)",
    )
    p.add_argument("--max-frames", type=int, default=None, help="Stop after N frames")
    p.add_argument("--equity-every-ms", type=int, default=1000, help="Equity sampling interval (ms)")
    p.add_argument("--progress-every-s", type=float, default=3.0, help="Console progress interval (seconds)")
    p.add_argument("--t0-ms", type=int, default=None, help="Start timestamp inclusive filter (ms)")
    p.add_argument("--t1-ms", type=int, default=None, help="End timestamp exclusive filter (ms)")
    p.add_argument(
        "--write-outputs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write trades.csv/orders.csv/equity.csv/summary.json (disable for sweeps)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    summary = asyncio.run(
        run_backtest(
            args.log,
            args.out,
            start_balance_usdt=args.start_balance,
            synth_fill_from_book=args.synth_fill_from_book,
            queue_ahead_factor=args.queue_ahead_factor,
            max_frames=args.max_frames,
            equity_every_ms=args.equity_every_ms,
            progress_every_s=args.progress_every_s,
            t0_ms=args.t0_ms,
            t1_ms=args.t1_ms,
            write_outputs=args.write_outputs,
        )
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

