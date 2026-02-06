from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

import websockets

from ..config import Settings
from ..utils.time_ms import now_ms

log = logging.getLogger("binance.ws_market")


@dataclass
class WSStats:
    connected: bool = False
    reconnects: int = 0
    last_event_ts_ms: int = 0


class BinanceMarketWS:
    """Market data WS (combined streams) for Binance USD-M Futures market data base."""

    def __init__(self, settings: Settings) -> None:
        self.s = settings
        self.stats = WSStats()

        # user-provided callbacks
        self.on_depth: Optional[Callable[[dict], Awaitable[None]]] = None
        self.on_trade: Optional[Callable[[dict], Awaitable[None]]] = None
        self.on_mark: Optional[Callable[[dict], Awaitable[None]]] = None

        self._task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()

    def _now_ms(self) -> int:
        return int(now_ms())

    def start(self) -> None:
        if self._task and not self._task.done():
            return
        self._stop.clear()
        self._task = asyncio.create_task(self._run(), name="ws_market")

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except Exception:
                pass

    def _build_url(self) -> str:
        sym = self.s.SYMBOL.lower()
        streams = [
            self.s.STREAMS_DEPTH.format(symbol_lower=sym),
            self.s.STREAMS_TRADES.format(symbol_lower=sym),
            self.s.STREAMS_MARK.format(symbol_lower=sym),
        ]
        return f"{self.s.WS_BASE}/stream?streams={'/'.join(streams)}"

    async def _dispatch(self, data: dict) -> None:
        # Combined stream format: {"stream": "...", "data": {...}}
        payload = data.get("data", data)
        stream = data.get("stream", "")

        if "depth" in stream and self.on_depth:
            await self.on_depth(payload)
        elif "aggTrade" in stream and self.on_trade:
            await self.on_trade(payload)
        elif "markPrice" in stream and self.on_mark:
            await self.on_mark(payload)

    async def _run(self) -> None:
        url = self._build_url()
        backoff = 0.5
        while not self._stop.is_set():
            try:
                async with websockets.connect(url, ping_interval=15, ping_timeout=15, close_timeout=2) as ws:
                    self.stats.connected = True
                    self.stats.last_event_ts_ms = self._now_ms()
                    backoff = 0.5
                    log.info("ws_connected %s", url)

                    while not self._stop.is_set():
                        msg = await ws.recv()
                        self.stats.last_event_ts_ms = self._now_ms()
                        data = json.loads(msg)
                        await self._dispatch(data)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.stats.connected = False
                self.stats.reconnects += 1
                log.warning("ws_reconnect err=%s", e)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 1.7, 8.0)

        self.stats.connected = False
        log.info("ws_stopped")

