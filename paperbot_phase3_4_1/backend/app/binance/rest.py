from __future__ import annotations

import aiohttp
import logging

from ..config import Settings

log = logging.getLogger("binance.rest")


class BinanceRest:
    """Minimal REST client for Binance USD-M Futures TESTNET."""

    def __init__(self, settings: Settings) -> None:
        self.s = settings
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=10)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def get_exchange_info(self) -> dict:
        sess = await self._get_session()
        url = f"{self.s.REST_BASE}/fapi/v1/exchangeInfo"
        async with sess.get(url) as r:
            r.raise_for_status()
            return await r.json()

    async def get_tick_size(self, symbol: str) -> float:
        info = await self.get_exchange_info()
        for s in info.get("symbols", []):
            if s.get("symbol") == symbol:
                for f in s.get("filters", []):
                    if f.get("filterType") == "PRICE_FILTER":
                        return float(f.get("tickSize", "0.1"))
        return 0.1

    async def get_depth_snapshot(self, symbol: str, limit: int = 1000) -> dict:
        sess = await self._get_session()
        url = f"{self.s.REST_BASE}/fapi/v1/depth"
        params = {"symbol": symbol, "limit": int(limit)}
        async with sess.get(url, params=params) as r:
            r.raise_for_status()
            return await r.json()

