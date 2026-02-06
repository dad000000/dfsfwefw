from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from typing import Optional

from ..binance.rest import BinanceRest
from ..binance.ws_market import BinanceMarketWS
from ..config import Settings
from ..paper.matching_engine import PaperMatchingEngine
from ..state.models import TradePrint
from ..state.store import Store
from ..utils.time_ms import now_ms
from ..utils.ndjson_recorder import NDJSONRecorder
from .risk import RiskManager
from .strategy import MicroScalpStrategy

log = logging.getLogger("bot.runner")


class BotRunner:
    def __init__(self, store: Store, rest: BinanceRest, ws: BinanceMarketWS, settings: Settings) -> None:
        self.store = store
        self.rest = rest
        self.ws = ws
        self.s = settings

        self.engine = PaperMatchingEngine(store, settings)
        self.risk = RiskManager(store, settings)

        # Phase 2.1: market+decisions recorder (NDJSON)
        rec_path = NDJSONRecorder.make_default_path(getattr(settings, "RECORDER_DIR", "recordings"), settings.SYMBOL)
        self.recorder = NDJSONRecorder(
            rec_path,
            enabled=bool(getattr(settings, "RECORDER_ENABLED", True)),
            flush_ms=int(getattr(settings, "RECORDER_FLUSH_MS", 2000)),
            queue_max=int(getattr(settings, "RECORDER_QUEUE_MAX", 5000)),
            rotate_max_mb=int(getattr(settings, "RECORDER_MAX_FILE_MB", 256)),
        )

        self.strategy = MicroScalpStrategy(store, self.engine, settings, recorder=self.recorder)

        self._risk_task: Optional[asyncio.Task] = None
        self._stop_all = asyncio.Event()

        # Depth snapshot+delta stitching helpers.
        # We buffer depth events so that when we fetch a snapshot we can
        # deterministically apply the correct deltas (Binance recommended flow).
        self._depth_buf = deque(maxlen=6000)  # ~10 minutes at 10Hz is plenty
        self._snapshot_lock = asyncio.Lock()
        self._drain_lock = asyncio.Lock()
        self._last_snapshot_ms: int = 0
        self._min_snapshot_interval_ms: int = 1500
        self._resync_requested: bool = False
        self._resync_task: Optional[asyncio.Task] = None
        self._consecutive_gap: int = 0

        self.ws.on_depth = self._on_depth
        self.ws.on_trade = self._on_trade
        self.ws.on_mark = self._on_mark

    async def start(self) -> None:
        await self.store.clear_kill()
        await self.store.set_bot_running(True)
        self._stop_all.clear()
        self._depth_buf.clear()
        self._consecutive_gap = 0
        await self.recorder.start()
        self.ws.start()
        if not self._risk_task or self._risk_task.done():
            self._risk_task = asyncio.create_task(self._risk_loop(), name="risk_loop")
        self.strategy.start()

    async def stop(self) -> None:
        await self.store.set_bot_running(False)
        await self.strategy.stop()
        await self.engine.cancel_all()
        await self.ws.stop()
        await self.recorder.stop()
        self._stop_all.set()
        if self._resync_task and not self._resync_task.done():
            self._resync_task.cancel()

    async def kill(self, reason: str) -> None:
        await self.store.trigger_kill(reason)
        await self.strategy.stop()
        await self.engine.cancel_all()

    async def reset(self) -> None:
        await self.stop()
        await self.store.reset_session()
        await self.risk.reset_daily()

    async def _risk_loop(self) -> None:
        while not self._stop_all.is_set():
            try:
                await self.store.update_ws_status(
                    self.ws.stats.connected,
                    self.ws.stats.reconnects,
                    self.ws.stats.last_event_ts_ms,
                )

                snap = await self.store.snapshot()
                if snap["connectivity"]["bot_running"] and not snap["connectivity"]["kill_switch"]:
                    st = await self.risk.check()
                    if not st.ok:
                        # STALE_BOOK is often recoverable (temporary gaps/resync).
                        if st.reason.startswith("STALE_BOOK"):
                            log.warning("risk_stale_book_resync reason=%s", st.reason)
                            await self._request_resync(force=True)
                        else:
                            log.warning("risk_kill_switch reason=%s", st.reason)
                            await self.kill(st.reason)
            except Exception as e:
                log.exception("risk_loop_error: %s", e)
            await asyncio.sleep(0.2)

    def _now_ms(self) -> int:
        return int(now_ms())

    async def _request_resync(self, force: bool = False) -> None:
        """Request a snapshot resync.

        We throttle resyncs to avoid hammering REST and blocking WS.
        """
        now = self._now_ms()
        if not force and (now - self._last_snapshot_ms) < self._min_snapshot_interval_ms:
            self._resync_requested = True
            return
        self._resync_requested = True
        if not self._resync_task or self._resync_task.done():
            self._resync_task = asyncio.create_task(self._resync_once(), name="depth_resync")

    async def _resync_once(self) -> None:
        async with self._snapshot_lock:
            if not self._resync_requested:
                return
            self._resync_requested = False

            now = self._now_ms()
            if (now - self._last_snapshot_ms) < self._min_snapshot_interval_ms:
                # still too soon; re-request and bail
                self._resync_requested = True
                return

            try:
                snap = await self.rest.get_depth_snapshot(self.s.SYMBOL, limit=1000)
                last_id = int(snap["lastUpdateId"])
                bids = [(float(p), float(q)) for p, q in snap.get("bids", [])]
                asks = [(float(p), float(q)) for p, q in snap.get("asks", [])]
                await self.store.set_book_from_snapshot(last_id, bids, asks)
                self._last_snapshot_ms = now
                self._consecutive_gap = 0
            except Exception as e:
                log.warning("depth_snapshot_failed err=%s", e)

        # after snapshot, immediately try to drain buffered deltas
        try:
            async with self._drain_lock:
                await self._drain_depth_buffer()
        except Exception as e:
            log.warning("drain_after_snapshot_failed err=%s", e)

    async def _drain_depth_buffer(self) -> None:
        """Apply buffered depth deltas in order using Binance stitching rules."""
        # We rely on Store.apply_depth_delta for the actual application,
        # but we handle discarding/initial alignment here so we don't thrash snapshots.
        while True:
            if not self._depth_buf:
                return

            # Need a snapshot first.
            if self.store.book.last_update_id == 0:
                await self._request_resync(force=True)
                return

            last_id = int(self.store.book.last_update_id)

            # Phase 1: align first delta after snapshot.
            if not self.store.book.depth_seq_started:
                # Drop anything older/equal to snapshot.
                while self._depth_buf and int(self._depth_buf[0].get("u", 0)) <= last_id:
                    self._depth_buf.popleft()
                if not self._depth_buf:
                    return

                U0 = int(self._depth_buf[0].get("U", 0))
                u0 = int(self._depth_buf[0].get("u", 0))

                # If the first buffered event starts after last_id+1, we missed the bridge.
                if U0 > last_id + 1:
                    self._consecutive_gap += 1
                    if self._consecutive_gap >= 2:
                        async with self.store.lock:
                            self.store.book.resync_failures += 1
                        await self._request_resync(force=True)
                    return

                # Discard until we find an event that bridges last_id+1.
                while self._depth_buf:
                    U = int(self._depth_buf[0].get("U", 0))
                    u = int(self._depth_buf[0].get("u", 0))
                    if u <= last_id:
                        self._depth_buf.popleft()
                        continue
                    if U <= last_id + 1 <= u:
                        break
                    # If U <= last_id+1 but u < last_id+1 -> too old, discard.
                    self._depth_buf.popleft()
                if not self._depth_buf:
                    return

            # Phase 2: apply as many sequential deltas as we can.
            applied_any = False
            while self._depth_buf:
                ev = self._depth_buf[0]
                event_ts = int(ev.get("E", self._now_ms()))
                U = int(ev["U"])
                u = int(ev["u"])
                b = [(float(p), float(q)) for p, q in ev.get("b", [])]
                a = [(float(p), float(q)) for p, q in ev.get("a", [])]

                ok = await self.store.apply_depth_delta(event_ts, U, u, b, a)
                if not ok:
                    # If we are already sequencing, any mismatch implies a gap -> resync.
                    self._consecutive_gap += 1
                    if self._consecutive_gap >= 2:
                        async with self.store.lock:
                            self.store.book.resync_failures += 1
                        await self._request_resync()
                    return

                self._depth_buf.popleft()
                applied_any = True
                self._consecutive_gap = 0

            if not applied_any:
                return

            # loop again in case more items arrived during awaits

    async def _on_depth(self, d: dict) -> None:
        try:
            # Buffer depth events and apply using snapshot+delta stitching.
            self._depth_buf.append(d)

            if self.store.book.last_update_id == 0:
                await self._request_resync(force=True)

            async with self._drain_lock:
                await self._drain_depth_buffer()

            await self.engine.on_book_tick()

        except Exception as e:
            log.exception("on_depth_error: %s", e)

    async def _on_trade(self, d: dict) -> None:
        try:
            ts = int(d.get("T", int(now_ms())))
            price = float(d["p"])
            qty = float(d["q"])
            is_bm = bool(d.get("m", False))
            side = "SELL" if is_bm else "BUY"

            t = TradePrint(ts_ms=ts, price=price, qty=qty, is_buyer_maker=is_bm, side=side)
            await self.store.on_trade_print(t)

            await self.engine.on_market_trade(t)

        except Exception as e:
            log.exception("on_trade_error: %s", e)

    async def _on_mark(self, d: dict) -> None:
        try:
            ts = int(d.get("E", int(now_ms())))
            mp = float(d["p"])
            await self.store.on_mark(ts, mp)
            await self.engine.on_book_tick()
        except Exception as e:
            log.exception("on_mark_error: %s", e)

